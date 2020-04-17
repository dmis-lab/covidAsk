import collections
import json
import logging
import os
import shutil
import torch
import math
import pandas as pd
import numpy as np
import six
from scipy.sparse import csr_matrix, save_npz, hstack, vstack
from termcolor import colored, cprint
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from eval_utils import normalize_answer, f1_score, exact_match_score

import h5py
from time import time
from multiprocessing import Queue, Process
from multiprocessing.pool import ThreadPool
from threading import Thread
from tqdm import tqdm as tqdm_
from decimal import *

import tokenization

QuestionResult = collections.namedtuple("QuestionResult",
                                        ['qas_id', 'start', 'end', 'sparse', 'input_ids'])

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def tqdm(*args, mininterval=5.0, **kwargs):
    return tqdm_(*args, mininterval=mininterval, **kwargs)


def get_metadata(id2example, features, results, max_answer_length, do_lower_case, verbose_logging):
    start = np.concatenate([result.start[1:len(feature.tokens) - 1] for feature, result in zip(features, results)],
                           axis=0)
    end = np.concatenate([result.end[1:len(feature.tokens) - 1] for feature, result in zip(features, results)], axis=0)

    input_ids = None
    sparse_map = None
    sparse_bi_map = None
    len_per_para = []
    if results[0].start_sp is not None:
        input_ids = np.concatenate([f.input_ids[1:len(f.tokens) - 1] for f in features], axis=0)
        sparse_features = None # uni
        sparse_bi_features = None
        if '1' in results[0].start_sp:
            sparse_features = [result.start_sp['1'][1:len(feature.tokens)-1, 1:len(feature.tokens)-1]
                               for feature, result in zip(features, results)]
        if '2' in results[0].start_sp:
            sparse_bi_features = [result.start_sp['2'][1:len(feature.tokens)-1, 1:len(feature.tokens)-1]
                               for feature, result in zip(features, results)]

        map_size = max([k.shape[0] for k in sparse_features])
        sparse_map = np.zeros((input_ids.shape[0], map_size), dtype=np.float32)
        if sparse_bi_features is not None:
            sparse_bi_map = np.zeros((input_ids.shape[0], map_size), dtype=np.float32)

        curr_size = 0
        for sidx, sparse_feature in enumerate(sparse_features):
            sparse_map[curr_size:curr_size + sparse_feature.shape[0],:sparse_feature.shape[1]] += sparse_feature
            if sparse_bi_features is not None:
                assert sparse_bi_features[sidx].shape == sparse_feature.shape
                sparse_bi_map[curr_size:curr_size+sparse_bi_features[sidx].shape[0],:sparse_bi_features[sidx].shape[1]] += \
                    sparse_bi_features[sidx]
            curr_size += sparse_feature.shape[0]
            len_per_para.append(sparse_feature.shape[0])

        assert input_ids.shape[0] == start.shape[0] and curr_size == sparse_map.shape[0]

    fs = np.concatenate([result.filter_start_logits[1:len(feature.tokens) - 1]
                         for feature, result in zip(features, results)],
                         axis=0)
    fe = np.concatenate([result.filter_end_logits[1:len(feature.tokens) - 1]
                         for feature, result in zip(features, results)],
                        axis=0)

    span_logits = np.zeros([np.shape(start)[0], max_answer_length], dtype=start.dtype)
    start2end = -1 * np.ones([np.shape(start)[0], max_answer_length], dtype=np.int32)
    idx = 0
    for feature, result in zip(features, results):
        for i in range(1, len(feature.tokens) - 1):
            for j in range(i, min(i + max_answer_length, len(feature.tokens) - 1)):
                span_logits[idx, j - i] = result.span_logits[i, j]
                start2end[idx, j - i] = idx + j - i
            idx += 1

    word2char_start = np.zeros([start.shape[0]], dtype=np.int32)
    word2char_end = np.zeros([start.shape[0]], dtype=np.int32)

    sep = ' [PAR] '
    full_text = ""
    prev_example = None

    word_pos = 0
    for feature in features:
        example = id2example[feature.unique_id]
        if prev_example is not None and feature.doc_span_index == 0:
            full_text = full_text + ' '.join(prev_example.doc_words) + sep

        for i in range(1, len(feature.tokens) - 1):
            _, start_pos, _ = get_final_text_(example, feature, i, min(len(feature.tokens) - 2, i + 1), do_lower_case,
                                              verbose_logging)
            _, _, end_pos = get_final_text_(example, feature, max(1, i - 1), i, do_lower_case,
                                            verbose_logging)
            start_pos += len(full_text)
            end_pos += len(full_text)
            word2char_start[word_pos] = start_pos
            word2char_end[word_pos] = end_pos
            word_pos += 1
        prev_example = example
    full_text = full_text + ' '.join(prev_example.doc_words)

    metadata = {'did': prev_example.doc_idx, 'context': full_text, 'title': prev_example.title,
                'start': start, 'end': end, 'span_logits': span_logits,
                'start2end': start2end,
                'word2char_start': word2char_start, 'word2char_end': word2char_end,
                'filter_start': fs, 'filter_end': fe, 'input_ids': input_ids,
                'sparse': sparse_map, 'sparse_bi': sparse_bi_map,
                'len_per_para': len_per_para}

    return metadata


def filter_metadata(metadata, threshold):
    start_idxs, = np.where(metadata['filter_start'] > threshold)
    end_idxs, = np.where(metadata['filter_end'] > threshold)
    end_long2short = {long: short for short, long in enumerate(end_idxs)}

    metadata['start'] = metadata['start'][start_idxs]
    metadata['end'] = metadata['end'][end_idxs]
    metadata['sparse'] = metadata['sparse'][start_idxs]
    if metadata['sparse_bi'] is not None:
        metadata['sparse_bi'] = metadata['sparse_bi'][start_idxs]
    metadata['f2o_start'] = start_idxs
    metadata['f2o_end'] = end_idxs
    metadata['span_logits'] = metadata['span_logits'][start_idxs]
    metadata['start2end'] = metadata['start2end'][start_idxs]
    for i, each in enumerate(metadata['start2end']):
        for j, long in enumerate(each.tolist()):
            metadata['start2end'][i, j] = end_long2short[long] if long in end_long2short else -1

    return metadata


def compress_metadata(metadata, dense_offset, dense_scale, sparse_offset, sparse_scale):
    for key in ['start', 'end']:
        if key in metadata:
            metadata[key] = float_to_int8(metadata[key], dense_offset, dense_scale)

    for key in ['sparse', 'sparse_bi']:
        if key in metadata and metadata[key] is not None:
            metadata[key] = float_to_int8(metadata[key], sparse_offset, sparse_scale)
    return metadata


def pool_func(item):
    metadata_ = get_metadata(*item[:-1])
    metadata_ = filter_metadata(metadata_, item[-1])
    return metadata_


def write_hdf5(all_examples, all_features, all_results,
               max_answer_length, do_lower_case, hdf5_path, filter_threshold, verbose_logging,
               dense_offset=None, dense_scale=None, sparse_offset=None, sparse_scale=None, use_sparse=False):
    assert len(all_examples) > 0


    id2feature = {feature.unique_id: feature for feature in all_features}
    id2example = {id_: all_examples[id2feature[id_].example_index] for id_ in id2feature}

    def add(inqueue_, outqueue_):
        for item in iter(inqueue_.get, None):
            args = list(item[:3]) + [max_answer_length, do_lower_case, verbose_logging, filter_threshold]
            out = pool_func(args)
            outqueue_.put(out)

        outqueue_.put(None)

    def write(outqueue_):
        with h5py.File(hdf5_path) as f:
            while True:
                metadata = outqueue_.get()
                if metadata:
                    did = str(metadata['did'])
                    if did in f:
                        logger.info('%s exists; replacing' % did)
                        del f[did]
                    dg = f.create_group(did)

                    dg.attrs['context'] = metadata['context']
                    dg.attrs['title'] = metadata['title']
                    if dense_offset is not None:
                        metadata = compress_metadata(metadata, dense_offset, dense_scale, sparse_offset, sparse_scale)
                        dg.attrs['offset'] = dense_offset
                        dg.attrs['scale'] = dense_scale
                        dg.attrs['sparse_offset'] = sparse_offset
                        dg.attrs['sparse_scale'] = sparse_scale
                    dg.create_dataset('start', data=metadata['start'])
                    dg.create_dataset('end', data=metadata['end'])
                    if metadata['sparse'] is not None:
                        dg.create_dataset('sparse', data=metadata['sparse'])
                        if metadata['sparse_bi'] is not None:
                            dg.create_dataset('sparse_bi', data=metadata['sparse_bi'])
                        dg.create_dataset('input_ids', data=metadata['input_ids'])
                        dg.create_dataset('len_per_para', data=metadata['len_per_para'])
                    dg.create_dataset('span_logits', data=metadata['span_logits'])
                    dg.create_dataset('start2end', data=metadata['start2end'])
                    dg.create_dataset('word2char_start', data=metadata['word2char_start'])
                    dg.create_dataset('word2char_end', data=metadata['word2char_end'])
                    dg.create_dataset('f2o_start', data=metadata['f2o_start'])
                    dg.create_dataset('f2o_end', data=metadata['f2o_end'])

                else:
                    break

    features = []
    results = []
    inqueue = Queue(maxsize=500)
    outqueue = Queue(maxsize=500)
    write_p = Thread(target=write, args=(outqueue,))
    p = Thread(target=add, args=(inqueue, outqueue))
    write_p.start()
    p.start()

    start_time = time()
    for count, result in enumerate(tqdm(all_results, total=len(all_features))):
        example = id2example[result.unique_id]
        feature = id2feature[result.unique_id]
        condition = len(features) > 0 and example.par_idx == 0 and feature.doc_span_index == 0

        if condition:
            in_ = (id2example, features, results)
            logger.info('inqueue size: %d, outqueue size: %d' % (inqueue.qsize(), outqueue.qsize()))
            inqueue.put(in_)
            # add(id2example, features, results)
            features = [feature]
            results = [result]
        else:
            features.append(feature)
            results.append(result)
        if count % 500 == 0:
            logger.info('%d/%d at %.1f' % (count + 1, len(all_features), time() - start_time))
    in_ = (id2example, features, results)
    inqueue.put(in_)
    inqueue.put(None)
    p.join()
    write_p.join()


def get_question_results(question_examples, query_eval_features, question_dataloader, device, model):
    id2feature = {feature.unique_id: feature for feature in query_eval_features}
    id2example = {id_: question_examples[id2feature[id_].example_index] for id_ in id2feature}
    for (input_ids_, input_mask_, example_indices) in question_dataloader:
        input_ids_ = input_ids_.to(device)
        input_mask_ = input_mask_.to(device)
        with torch.no_grad():
            batch_start, batch_end, batch_sps, batch_eps = model(query_ids=input_ids_,
                                                                 query_mask=input_mask_)
        for i, example_index in enumerate(example_indices):
            start = batch_start[i].detach().cpu().numpy().astype(np.float16)
            end = batch_end[i].detach().cpu().numpy().astype(np.float16)
            sparse = None
            if len(batch_sps) > 0:
                sparse = {ng: bb_ssp[i].detach().cpu().numpy().astype(np.float16) for ng, bb_ssp in batch_sps.items()}
            query_eval_feature = query_eval_features[example_index.item()]
            unique_id = int(query_eval_feature.unique_id)
            qas_id = id2example[unique_id].qas_id
            yield QuestionResult(qas_id=qas_id,
                                 start=start,
                                 end=end,
                                 sparse=sparse,
                                 input_ids=query_eval_feature.input_ids[1:len(query_eval_feature.tokens_)-1])


def convert_question_features_to_dataloader(query_eval_features, fp16, local_rank, predict_batch_size):
    all_input_ids_ = torch.tensor([f.input_ids for f in query_eval_features], dtype=torch.long)
    all_input_mask_ = torch.tensor([f.input_mask for f in query_eval_features], dtype=torch.long)
    all_example_index_ = torch.arange(all_input_ids_.size(0), dtype=torch.long)
    if fp16:
        all_input_ids_, all_input_mask_ = tuple(t.half() for t in (all_input_ids_, all_input_mask_))

    question_data = TensorDataset(all_input_ids_, all_input_mask_, all_example_index_)

    if local_rank == -1:
        question_sampler = SequentialSampler(question_data)
    else:
        question_sampler = DistributedSampler(question_data)
    question_dataloader = DataLoader(question_data, sampler=question_sampler, batch_size=predict_batch_size)
    return question_dataloader


def get_final_text_(example, feature, start_index, end_index, do_lower_case, verbose_logging):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    orig_doc_start = feature.token_to_word_map[start_index]
    orig_doc_end = feature.token_to_word_map[end_index]
    orig_words = example.doc_words[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_words)
    full_text = " ".join(example.doc_words)

    start_pos, end_pos = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging) # TODO: need to check
    offset = sum(len(word) + 1 for word in example.doc_words[:orig_doc_start])

    return full_text, offset + start_pos, offset + end_pos


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.
    default_out = 0, len(orig_text)

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return default_out
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return default_out

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return default_out

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return default_out

    # output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return orig_start_position, orig_end_position + 1


def float_to_int8(num, offset, factor, keep_zeros=False):
    out = (num - offset) * factor
    out = out.clip(-128, 127)
    if keep_zeros:
        out = out * (num != 0.0).astype(np.int8)
    out = np.round(out).astype(np.int8)
    return out


def int8_to_float(num, offset, factor, keep_zeros=False):
    if not keep_zeros:
        return num.astype(np.float32) / factor + offset
    else:
        return (num.astype(np.float32) / factor + offset) * (num != 0.0).astype(np.float32)
