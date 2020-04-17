import argparse
import json
import os
import random
import logging
from collections import namedtuple, Counter
from time import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

from scipy.sparse import vstack

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MIPS(object):
    def __init__(self, phrase_dump_dir, tfidf_dump_dir, start_index_path, idx2id_path, max_norm_path,
                 doc_rank_fn, cuda=False, dump_only=False):

        # If dump dir is a file, use it as a dump.
        if os.path.isdir(phrase_dump_dir):
            self.phrase_dump_paths = sorted(
                [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir) if 'hdf5' in name]
            )
            dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.phrase_dump_paths]
            self.dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
        else:
            self.phrase_dump_paths = [phrase_dump_dir]
        self.phrase_dumps = [h5py.File(path, 'r') for path in self.phrase_dump_paths]

        # Load tfidf dump
        assert os.path.isdir(tfidf_dump_dir), tfidf_dump_dir
        self.tfidf_dump_paths = sorted(
            [os.path.join(tfidf_dump_dir, name) for name in os.listdir(tfidf_dump_dir) if 'hdf5' in name]
        )
        tfidf_dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.tfidf_dump_paths]
        if '-' in tfidf_dump_names[0]: # Range check
            tfidf_dump_ranges = [list(map(int, name.split('_')[0].split('-'))) for name in tfidf_dump_names]
            assert tfidf_dump_ranges == self.dump_ranges
        self.tfidf_dumps = [h5py.File(path, 'r') for path in self.tfidf_dump_paths]
        logger.info(f'using doc ranker functions: {doc_rank_fn["index"]}')
        self.doc_rank_fn = doc_rank_fn
        if dump_only:
            return

        # Read index
        logger.info(f'Reading {start_index_path}')
        import faiss
        self.start_index = faiss.read_index(start_index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        self.idx_f = self.load_idx_f(idx2id_path)
        with open(max_norm_path, 'r') as fp:
            self.max_norm = json.load(fp)

        # Options
        self.num_docs_list = []
        self.cuda = cuda
        if self.cuda:
            assert torch.cuda.is_available(), f"Cuda availability {torch.cuda.is_available()}"
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

    def close(self):
        for phrase_dump in self.phrase_dumps:
            phrase_dump.close()
        for tfidf_dump in self.tfidf_dumps:
            tfidf_dump.close()

    def load_idx_f(self, idx2id_path):
        idx_f = {}
        types = ['doc', 'word']
        with h5py.File(idx2id_path, 'r', driver='core', backing_store=False) as f:
            for key in tqdm(f, desc='loading idx2id'):
                idx_f_cur = {}
                for type_ in types:
                    idx_f_cur[type_] = f[key][type_][:]
                idx_f[key] = idx_f_cur
            return idx_f

    def get_idxs(self, I):
        offsets = (I / 1e8).astype(np.int64) * int(1e8)
        idxs = I % int(1e8)
        doc = np.array(
            [[self.idx_f[str(offset)]['doc'][idx] for offset, idx in zip(oo, ii)] for oo, ii in zip(offsets, idxs)])
        word = np.array([[self.idx_f[str(offset)]['word'][idx] for offset, idx in zip(oo, ii)] for oo, ii in
                         zip(offsets, idxs)])
        return doc, word

    def get_doc_group(self, doc_idx):
        if len(self.phrase_dumps) == 1:
            return self.phrase_dumps[0][str(doc_idx)]
        for dump_range, dump in zip(self.dump_ranges, self.phrase_dumps):
            if dump_range[0] * 1000 <= int(doc_idx) < dump_range[1] * 1000:
                if str(doc_idx) not in dump:
                    raise ValueError('%d not found in dump list' % int(doc_idx))
                return dump[str(doc_idx)]

        # Just check last
        if str(doc_idx) not in self.phrase_dumps[-1]:
            raise ValueError('%d not found in dump list' % int(doc_idx))
        else:
            return self.phrase_dumps[-1][str(doc_idx)]

    def get_tfidf_group(self, doc_idx):
        if len(self.tfidf_dumps) == 1:
            return self.tfidf_dumps[0][str(doc_idx)]
        for dump_range, dump in zip(self.dump_ranges, self.tfidf_dumps):
            if dump_range[0] * 1000 <= int(doc_idx) < dump_range[1] * 1000:
                return dump[str(doc_idx)]

        # Just check last
        if str(doc_idx) not in self.tfidf_dumps[-1]:
            raise ValueError('%d not found in dump list' % int(doc_idx))
        else:
            return self.tfidf_dumps[-1][str(doc_idx)]

    def int8_to_float(self, num, offset, factor):
        return num.astype(np.float32) / factor + offset

    def adjust(self, each):
        last = each['context'].rfind(' [PAR] ', 0, each['start_pos'])
        last = 0 if last == -1 else last + len(' [PAR] ')
        next = each['context'].find(' [PAR] ', each['end_pos'])
        next = len(each['context']) if next == -1 else next
        each['context'] = each['context'][last:next]
        each['start_pos'] -= last
        each['end_pos'] -= last
        return each

    def scale_l2_to_ip(self, l2_scores, max_norm=None, query_norm=None):
        """
        sqrt(m^2 + q^2 - 2qx) -> m^2 + q^2 - 2qx -> qx - 0.5 (q^2 + m^2)
        Note that faiss index returns squared euclidean distance, so no need to square it again.
        """
        if max_norm is None:
            return -0.5 * l2_scores
        assert query_norm is not None
        return -0.5 * (l2_scores - query_norm ** 2 - max_norm ** 2)

    def dequant(self, group, input_, attr='dense'):
        # return input_

        if 'offset' not in group.attrs:
            return input_

        if attr == 'dense':
            return self.int8_to_float(input_, group.attrs['offset'], group.attrs['scale'])
        elif attr == 'sparse':
            return self.int8_to_float(input_, group.attrs['sparse_offset'], group.attrs['sparse_scale'])
        else:
            raise NotImplementedError()

    def sparse_bmm(self, q_ids, q_vals, p_ids, p_vals):
        """
        Efficient batch inner product after slicing (matrix x matrix)
        """
        q_max = max([len(q) for q in q_ids])
        p_max = max([len(p) for p in p_ids])
        factor = len(p_ids)//len(q_ids)
        assert q_max == max([len(q) for q in q_vals]) and p_max == max([len(p) for p in p_vals])
        with torch.no_grad():
            q_ids_pad = torch.LongTensor([q_id.tolist() + [0]*(q_max-len(q_id)) for q_id in q_ids]).to(self.device)
            q_ids_pad = q_ids_pad.repeat(1, factor).view(len(p_ids), -1) # Repeat for p
            q_vals_pad = torch.FloatTensor([q_val.tolist() + [0]*(q_max-len(q_val)) for q_val in q_vals]).to(self.device)
            q_vals_pad = q_vals_pad.repeat(1, factor).view(len(p_vals), -1) # Repeat for p
            p_ids_pad = torch.LongTensor([p_id.tolist() + [0]*(p_max-len(p_id)) for p_id in p_ids]).to(self.device)
            p_vals_pad = torch.FloatTensor([p_val.tolist() + [0]*(p_max-len(p_val)) for p_val in p_vals]).to(self.device)
            id_map = q_ids_pad.unsqueeze(1)
            id_map_ = p_ids_pad.unsqueeze(2)
            match = (id_map == id_map_).to(torch.float32)
            val_map = q_vals_pad.unsqueeze(1)
            val_map_ = p_vals_pad.unsqueeze(2)
            sp_scores = ((val_map * val_map_) * match).sum([1, 2])
        return sp_scores.cpu().numpy()

    def search_dense(self, q_texts, query_start, start_top_k, nprobe, sparse_weight=0.05):
        batch_size = query_start.shape[0]
        self.start_index.nprobe = nprobe

        # Query concatenation for l2 to ip
        query_start = np.concatenate([np.zeros([batch_size, 1]).astype(np.float32), query_start], axis=1)

        # Search with faiss
        start_scores, I = self.start_index.search(query_start, start_top_k)
        query_norm = np.linalg.norm(query_start, ord=2, axis=1)
        start_scores = self.scale_l2_to_ip(start_scores, max_norm=self.max_norm, query_norm=np.expand_dims(query_norm, 1))

        # Get idxs from resulting I
        doc_idxs, start_idxs = self.get_idxs(I)

        # For record
        num_docs = sum([len(set(doc_idx.flatten().tolist())) for doc_idx in doc_idxs]) / batch_size
        self.num_docs_list.append(num_docs)

        # Doc-level sparse score
        b_doc_scores = self.doc_rank_fn['index'](q_texts, doc_idxs.tolist()) # Index
        for b_idx in range(batch_size):
            start_scores[b_idx] += np.array(b_doc_scores[b_idx]) * sparse_weight

        return (doc_idxs, start_idxs), start_scores

    def search_sparse(self, q_texts, query_start, doc_top_k, start_top_k, sparse_weight=0.05):
        batch_size = query_start.shape[0]

        # Reduce search space by doc scores
        top_doc_idxs, top_doc_scores = self.doc_rank_fn['top_docs'](q_texts, doc_top_k) # Top docs

        # For each item, add start scores
        b_doc_idxs = []
        b_start_idxs = []
        b_scores = []
        max_phrases = 0
        for b_idx in range(batch_size):
            doc_idxs = []
            start_idxs = []
            scores = []
            for doc_idx, doc_score in zip(top_doc_idxs[b_idx], top_doc_scores[b_idx]):
                try:
                    doc_group = self.get_doc_group(doc_idx)
                except ValueError:
                    continue
                start = self.dequant(doc_group, doc_group['start'][:])
                cur_scores = np.sum(query_start[b_idx] * start, 1)
                for i, cur_score in enumerate(cur_scores):
                    doc_idxs.append(doc_idx)
                    start_idxs.append(i)
                    scores.append(cur_score + sparse_weight * doc_score)
            max_phrases = len(scores) if len(scores) > max_phrases else max_phrases

            b_doc_idxs.append(doc_idxs)
            b_start_idxs.append(start_idxs)
            b_scores.append(scores)
        # mean_val = [sum(scores)/len(scores) for scores in b_scores]
        # mean_val = sum(mean_val) / len(mean_val)
        mean_val = 0

        # If start_top_k is larger than nonnegative doc_idxs, we need to cut them later
        for doc_idxs, start_idxs, scores in zip(b_doc_idxs, b_start_idxs, b_scores):
            doc_idxs += [-1] * (max_phrases - len(doc_idxs))
            start_idxs += [-1] * (max_phrases - len(start_idxs))
            scores += [-10**9] * (max_phrases - len(scores))

        doc_idxs, start_idxs, scores = np.stack(b_doc_idxs), np.stack(b_start_idxs), np.stack(b_scores)
        return (doc_idxs, start_idxs), scores, mean_val

    def batch_par_scores(self, q_texts, q_sparses, doc_idxs, start_idxs, sparse_weight=0.05, mid_top_k=100):
        # Reshape for sparse
        num_queries = len(q_texts)
        doc_idxs = np.reshape(doc_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])

        default_doc = [doc_idx for doc_idx in doc_idxs if doc_idx >= 0][0]
        groups = [self.get_doc_group(doc_idx) if doc_idx >= 0 else self.get_doc_group(default_doc)
                  for doc_idx in doc_idxs]

        # Calculate paragraph start end location in sparse vector
        para_lens = [group['len_per_para'][:] for group in groups]
        f2o_start = [group['f2o_start'][:] for group in groups]
        para_bounds = [[(sum(para_len[:para_idx]), sum(para_len[:para_idx+1])) for
                        para_idx in range(len(para_len))] for para_len in para_lens]
        para_idxs = []
        for para_bound, start_idx, f2o in zip(para_bounds, start_idxs, f2o_start):
            para_bound = np.array(para_bound)
            curr_idx = ((f2o[start_idx] >= para_bound[:,0]) & (f2o[start_idx] < para_bound[:,1])).nonzero()[0][0]
            para_idxs.append(curr_idx)
        para_startend = [para_bound[para_idx] for para_bound, para_idx in zip(para_bounds, para_idxs)]

        # 1) TF-IDF based paragraph score
        q_spvecs = self.doc_rank_fn['spvec'](q_texts) # Spvec
        qtf_ids = [np.array(q) for q in q_spvecs[1]]
        qtf_vals = [np.array(q) for q in q_spvecs[0]]
        tfidf_groups = [self.get_tfidf_group(doc_idx) if doc_idx >= 0 else self.get_tfidf_group(default_doc)
                        for doc_idx in doc_idxs]
        tfidf_groups = [group[str(para_idx)] for group, para_idx in zip(tfidf_groups, para_idxs)]
        ptf_ids = [data['idxs'][:] for data in tfidf_groups]
        ptf_vals = [data['vals'][:] for data in tfidf_groups]
        tf_scores = self.sparse_bmm(qtf_ids, qtf_vals, ptf_ids, ptf_vals) * sparse_weight

        # 2) Sparse vectors based paragraph score
        q_ids, q_unis, q_bis = q_sparses
        q_ids = [np.array(q) for q in q_ids]
        q_unis = [np.array(q) for q in q_unis]
        q_bis = [np.array(q)[:-1] for q in q_bis]
        p_ids_tmp = [group['input_ids'][:] for group in groups]
        p_unis_tmp = [group['sparse'][:, :] for group in groups]
        p_bis_tmp = [group['sparse_bi'][:, :] for group in groups]
        p_ids = [sparse_id[p_se[0]:p_se[1]]
                 for sparse_id, p_se in zip(p_ids_tmp, para_startend)]
        p_unis = [self.dequant(groups[0], sparse_val[start_idx,:p_se[1]-p_se[0]], attr='sparse')
                for sparse_val, p_se, start_idx in zip(p_unis_tmp, para_startend, start_idxs)]
        p_bis = [self.dequant(groups[0], sparse_bi_val[start_idx,:p_se[1]-p_se[0]-1], attr='sparse')
                for sparse_bi_val, p_se, start_idx in zip(p_bis_tmp, para_startend, start_idxs)]
        sp_scores = self.sparse_bmm(q_ids, q_unis, p_ids, p_unis)

        # For bigram
        MAXV = 30522
        q_bids = [np.array([a*MAXV+b for a, b in zip(q_id[:-1], q_id[1:])]) for q_id in q_ids]
        p_bids = [np.array([a*MAXV+b for a, b in zip(p_id[:-1], p_id[1:])]) for p_id in p_ids]
        sp_scores += self.sparse_bmm(q_bids, q_bis, p_bids, p_bis)

        return np.reshape(tf_scores + sp_scores, [num_queries, -1])

    def search_start(self, query_start, sparse_query, q_texts=None,
                     nprobe=16, doc_top_k=5, start_top_k=100, mid_top_k=20, top_k=5,
                     search_strategy='dense_first', sparse_weight=0.05, no_para=False):

        assert self.start_index is not None
        query_start = query_start.astype(np.float32)
        batch_size = query_start.shape[0]
        # start_time = time()

        # 1) Branch based on the strategy (start_top_k) + doc_score
        if search_strategy == 'dense_first':
            (doc_idxs, start_idxs), start_scores = self.search_dense(
                q_texts, query_start, start_top_k, nprobe, sparse_weight
            )
        elif search_strategy == 'sparse_first':
            (doc_idxs, start_idxs), start_scores, _ = self.search_sparse(
                q_texts, query_start, doc_top_k, start_top_k, sparse_weight
            )
        elif search_strategy == 'hybrid':
            (doc_idxs, start_idxs), start_scores = self.search_dense(
                q_texts, query_start, start_top_k, nprobe, sparse_weight
            )
            (doc_idxs_, start_idxs_), start_scores_, sparse_mean = self.search_sparse(
                q_texts, query_start, doc_top_k, start_top_k, sparse_weight
            )

            # There could be a duplicate but it's difficult to remove
            doc_idxs = np.concatenate([doc_idxs, doc_idxs_], -1)
            start_idxs = np.concatenate([start_idxs, start_idxs_], -1)
            start_scores = np.concatenate([start_scores, start_scores_], -1)
        else:
            raise ValueError(search_strategy)

        # 2) Rerank and reduce (mid_top_k)
        rerank_idxs = np.argsort(start_scores, axis=1)[:,-mid_top_k:][:,::-1]
        doc_idxs = doc_idxs.tolist()
        start_idxs = start_idxs.tolist()
        start_scores = start_scores.tolist()
        for b_idx in range(batch_size):
            doc_idxs[b_idx] = np.array(doc_idxs[b_idx])[rerank_idxs[b_idx]]
            start_idxs[b_idx] = np.array(start_idxs[b_idx])[rerank_idxs[b_idx]]
            start_scores[b_idx] = np.array(start_scores[b_idx])[rerank_idxs[b_idx]]

        # logger.info(f'1st rerank ({start_top_k} => {mid_top_k}), {np.array(start_scores).shape}, {time()-start_time}')
        # start_time = time()

        # Para-level sparse score
        if not no_para:
            par_scores = self.batch_par_scores(q_texts, sparse_query, doc_idxs, start_idxs, sparse_weight, mid_top_k)
            start_scores = np.stack(start_scores) + par_scores
            start_scores = [s for s in start_scores]

        # 3) Rerank and reduce (top_k)
        rerank_idxs = np.argsort(start_scores, axis=1)[:,-top_k:][:,::-1]
        for b_idx in range(batch_size):
            doc_idxs[b_idx] = doc_idxs[b_idx][rerank_idxs[b_idx]]
            start_idxs[b_idx] = start_idxs[b_idx][rerank_idxs[b_idx]]
            start_scores[b_idx] = start_scores[b_idx][rerank_idxs[b_idx]]

        doc_idxs = np.stack(doc_idxs)
        start_idxs = np.stack(start_idxs)
        start_scores = np.stack(start_scores)

        # logger.info(f'2nd rerank ({mid_top_k} => {top_k}), {start_scores.shape}, {time()-start_time}')
        return start_scores, doc_idxs, start_idxs

    def search_end(self, query, doc_idxs, start_idxs, start_scores=None, top_k=5, max_answer_length=20):
        # Reshape for end
        num_queries = query.shape[0]
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        q_idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])
        start_scores = np.reshape(start_scores, [-1])

        # Get query_end and groups
        bs = int((query.shape[1] - 1) / 2) # Boundary of start
        query_end, query_span_logit = query[:,bs:2*bs], query[:,-1:]
        default_doc = [doc_idx for doc_idx in doc_idxs if doc_idx >= 0][0]
        groups = [self.get_doc_group(doc_idx) if doc_idx >= 0 else self.get_doc_group(default_doc)
                  for doc_idx in doc_idxs]
        ends = [group['end'][:] for group in groups]
        spans = [group['span_logits'][:] for group in groups]
        default_end = np.zeros(bs).astype(np.float32)

        # Calculate end
        end_idxs = [group['start2end'][start_idx, :max_answer_length]
            for group, start_idx in zip(groups, start_idxs)]  # [Q, L]
        end_mask = -1e9 * (np.array(end_idxs) < 0)  # [Q, L]

        end = np.stack([[each_end[each_end_idx, :] if each_end.size > 0 else default_end
                         for each_end_idx in each_end_idxs]
                        for each_end, each_end_idxs in zip(ends, end_idxs)], 0)  # [Q, L, d]
        end = self.dequant(groups[0], end)
        span = np.stack([[each_span[start_idx, i] for i in range(len(each_end_idxs))]
                         for each_span, start_idx, each_end_idxs in zip(spans, start_idxs, end_idxs)], 0)  # [Q, L]

        with torch.no_grad():
            end = torch.FloatTensor(end).to(self.device)
            query_end = torch.FloatTensor(query_end).to(self.device)
            end_scores = (query_end.unsqueeze(1) * end).sum(2).cpu().numpy()
        span_scores = query_span_logit * span  # [Q, L]
        scores = np.expand_dims(start_scores, 1) + end_scores + span_scores + end_mask  # [Q, L]
        pred_end_idxs = np.stack([each[idx] for each, idx in zip(end_idxs, np.argmax(scores, 1))], 0)  # [Q]
        max_scores = np.max(scores, 1)

        # Calculate doc_meta
        start_chars = [
            group['word2char_start'][group['f2o_start'][start_idx]].item() for start_idx, group in zip(start_idxs, groups)
        ]
        doc_metas = [self.doc_rank_fn['doc_meta'](group.attrs['title']) for group in groups]

        sent_start_pos = []
        sent_end_pos = []
        context_starts = []
        context_ends = []
        for start_char, doc_meta, group in zip(start_chars, doc_metas, groups):
            # TODO: assumes a single para in doc
            para_idx = 0
            if 'paragraphs' not in doc_meta:
                sent_start_pos.append(start_char)
                sent_end_pos.append(start_char)
                context_starts.append(0)
                context_ends.append(len(group.attrs['context']))
                continue
            sent_bounds = doc_meta['paragraphs'][para_idx]['context_sent_idx']
            for sent_idx, sent_bound in enumerate(sent_bounds):
                if start_char >= sent_bound[0] and start_char < sent_bound[1]:
                    sent_start_pos.append(sent_bound[0])
                    sent_end_pos.append(sent_bound[1])
                    before_idx = sent_idx - 1 if sent_idx > 0 else sent_idx
                    after_idx = (sent_idx + 1 if sent_idx < len(sent_bounds)-1 else sent_idx)
                    context_starts.append(sent_bounds[before_idx][0])
                    context_ends.append(sent_bounds[after_idx][1])
                    break

        # Get answers
        out = [{'context': group.attrs['context'], 'title': group.attrs['title'], 'doc_idx': doc_idx,
                'c_start': c_start, 'c_end': c_end if c_end <= len(group.attrs['context']) else len(group.attrs['context']),
                'sent_start': sent_start,
                'sent_end': sent_end if sent_end <= len(group.attrs['context']) else len(group.attrs['context']),
                'start_pos': group['word2char_start'][group['f2o_start'][start_idx]].item(),
                'end_pos': (group['word2char_end'][group['f2o_end'][end_idx]].item() if len(group['word2char_end']) > 0
                    else group['word2char_start'][group['f2o_start'][start_idx]].item() + 1),
                'start_idx': start_idx, 'end_idx': end_idx, 'score': score,
                'metadata': self.doc_rank_fn['doc_meta'](group.attrs['title'])}
               for doc_idx, group, start_idx, end_idx, score, c_start, c_end, sent_start, sent_end in zip(
                                                                    doc_idxs.tolist(), groups, start_idxs.tolist(),
                                                                    pred_end_idxs.tolist(), max_scores.tolist(),
                                                                    context_starts, context_ends,
                                                                    sent_start_pos, sent_end_pos)]
        for each in out:
            each['answer'] = each['context'][each['start_pos']:each['end_pos']]
        out = [self.adjust(each) for each in out]

        # Sort output
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(q_idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])
            new_out[i] = list(filter(lambda x: x['score'] > -1e5, new_out[i])) # In case of no output but masks
            new_out[i] = list(filter(lambda x: x['end_pos'] - x['start_pos'] + 1 < 500, new_out[i])) # filter long sents

        return new_out

    def filter_results(self, results):
        out = []
        for result in results:
            c = Counter(result['context'])
            if c['?'] > 3:
                continue
            if c['!'] > 5:
                continue
            out.append(result)
        return out

    # Use this only for demo (not good for performance evaluation)
    def aggregate_answers(self, batch_item):
        new_out = []
        for topk_item in batch_item:
            new_topk = {}
            for item in topk_item:
                doc_idx = str(item['doc_idx'])
                if doc_idx not in new_topk:
                    new_topk[doc_idx] = item
                else:
                    new_topk[doc_idx] = item if item['score'] > new_topk[doc_idx]['score'] else new_topk[doc_idx]
            new_out.append([it for it in new_topk.values()])

        # Re-sort
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])
        return new_out

    def search(self, query, sparse_query, q_texts=None,
               nprobe=256, doc_top_k=5, start_top_k=1000, mid_top_k=100, top_k=10,
               search_strategy='dense_first', filter_=False, aggregate=False, return_idxs=False,
               max_answer_length=20, sparse_weight=0.05, no_para=False):

        # Search start
        start_scores, doc_idxs, start_idxs = self.search_start(
            query[:, :int((query.shape[1] -1) / 2)],
            sparse_query,
            q_texts=q_texts,
            nprobe=nprobe,
            doc_top_k=doc_top_k,
            start_top_k=start_top_k,
            mid_top_k=mid_top_k,
            top_k=top_k,
            search_strategy=search_strategy,
            sparse_weight=sparse_weight,
            no_para=no_para,
        )

        # start_time = time()
        # Search end
        outs = self.search_end(
            query, doc_idxs, start_idxs, start_scores=start_scores,
            top_k=top_k, max_answer_length=max_answer_length
        )
        # logger.info(f'last rerank ({top_k}), {len(outs)}, {time()-start_time}')

        if filter_:
            outs = [self.filter_results(results) for results in outs]
        if aggregate:
            outs = self.aggregate_answers(outs)
        if return_idxs:
            return [[(out_['doc_idx'], out_['start_idx'], out_['end_idx'], out_['answer']) for out_ in out ] for out in outs]
        if doc_idxs.shape[1] != top_k:
            logger.info(f"Warning.. {doc_idxs.shape[1]} only retrieved")
            top_k = doc_idxs.shape[1]

        return outs
