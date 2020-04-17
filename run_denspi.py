# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import argparse
import collections
import logging
import json
import os
import random
import subprocess
from tqdm import tqdm as tqdm_
from time import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam

import tokenization
from modeling import BertConfig, DenSPI
from optimization import BERTAdam
from post import write_hdf5, convert_question_features_to_dataloader
from pre import convert_examples_to_features, read_squad_examples, convert_documents_to_features 


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

ContextResult = collections.namedtuple(
    "ContextResult",
    ['unique_id', 'start', 'end', 'span_logits', 'filter_start_logits', 'filter_end_logits', 'start_sp', 'end_sp']
)


def tqdm(*args, mininterval=5.0, **kwargs):
    return tqdm_(*args, mininterval=mininterval, **kwargs)


def main():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument("--predict_file", default='dev-v1.1.json', type=str, help="json for prediction.")

    # Metadata paths
    parser.add_argument('--metadata_dir', default='models/bert', type=str, help="Dir for pre-trained models.")
    parser.add_argument("--vocab_file", default='vocab.txt', type=str, help="Vocabulary file of pre-trained model.")
    parser.add_argument("--bert_model_option", default='large_uncased', type=str,
                        help="model architecture option. [large_uncased] or [base_uncased].")
    parser.add_argument("--bert_config_file", default='bert_config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT model.")
    parser.add_argument("--init_checkpoint", default='pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")

    # Output and load paths
    parser.add_argument("--output_dir", default='out/', type=str, help="storing models and predictions")
    parser.add_argument("--dump_dir", default='test/', type=str)
    parser.add_argument("--dump_file", default='phrase.hdf5', type=str, help="dump phrases of file.")
    parser.add_argument('--load_dir', default='out/', type=str, help="Dir for checkpoints of models to load.")
    parser.add_argument('--load_epoch', type=str, default='1', help="Epoch of model to load.")

    # Do's
    parser.add_argument("--do_load", default=False, action='store_true', help='Do load. If eval, do load automatically')
    parser.add_argument('--do_dump', default=False, action='store_true')

    # Model options: if you change these, you need to train again
    parser.add_argument("--do_case", default=False, action='store_true', help="Whether to keep upper casing")
    parser.add_argument("--use_sparse", default=False, action='store_true')
    parser.add_argument("--sparse_ngrams", default='1,2', type=str)
    parser.add_argument("--skip_no_answer", default=False, action='store_true')
    parser.add_argument('--freeze_word_emb', default=False, action='store_true')
    parser.add_argument('--append_title', default=False, action='store_true')

    # GPU and memory related options
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--predict_batch_size", default=64, type=int, help="Total batch size for predictions.")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    # Prediction options: only effective during prediction
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    # Index Options
    parser.add_argument('--dtype', default='float32', type=str)
    parser.add_argument('--filter_threshold', default=-1e9, type=float)
    parser.add_argument('--dense_offset', default=-2, type=float) # Original
    parser.add_argument('--dense_scale', default=20, type=float)
    parser.add_argument('--sparse_offset', default=1.6, type=float)
    parser.add_argument('--sparse_scale', default=80, type=float)

    # Others
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--seed', type=int, default=45,
                        help="random seed for initialization")
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--draft_num_examples', type=int, default=12)

    args = parser.parse_args()

    # Filesystem routines
    class Processor(object):
        def __init__(self, save_path, load_path):
            self._save = None
            self._load = None
            self._save_path = save_path
            self._load_path = load_path

        def bind(self, save, load):
            self._save = save
            self._load = load

        def save(self, checkpoint=None, save_fn=None, **kwargs):
            path = os.path.join(self._save_path, str(checkpoint))
            if save_fn is None:
                self._save(path, **kwargs)
            else:
                save_fn(path, **kwargs)

        def load(self, checkpoint, load_fn=None, session=None, **kwargs):
            assert self._load_path == session
            path = os.path.join(self._load_path, str(checkpoint), 'model.pt')
            if load_fn is None:
                self._load(path, **kwargs)
            else:
                load_fn(path, **kwargs)

    processor = Processor(args.output_dir, args.load_dir)
    if args.do_load is False:
        logger.info("Setting do_load to true for dumping")
        args.do_load = True

    # Configure file paths
    args.predict_file = os.path.join(args.data_dir, args.predict_file)
    args.vocab_file = os.path.join(args.metadata_dir, args.vocab_file)
    args.bert_config_file = os.path.join(
        args.metadata_dir, args.bert_config_file.replace(".json", "") + "_" + args.bert_model_option + ".json"
    )
    args.init_checkpoint = os.path.join(
        args.metadata_dir, args.init_checkpoint.replace(".bin", "") + "_" + args.bert_model_option + ".bin"
    )
    args.dump_file = os.path.join(args.dump_dir, args.dump_file)

    # CUDA Check
    logger.info('cuda availability: {}'.format(torch.cuda.is_available()))

    # Multi-GPU stuff
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Overwriting outputs in %s"% args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.dump_dir) and os.listdir(args.dump_dir):
        logger.info("Overwriting dump in %s"% args.dump_dir)
    else:
        os.makedirs(args.dump_dir, exist_ok=True)

    model = DenSPI(bert_config,
        sparse_ngrams=args.sparse_ngrams.split(','),
        use_sparse=args.use_sparse,
    )
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=not args.do_case)

    # Initialize BERT if not loading and has init_checkpoint
    if not args.do_load and args.init_checkpoint is not None:
        if args.draft:
            logger.info('[Draft] Randomly initialized model')
        else:
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            if next(iter(state_dict)).startswith('bert.'):
                state_dict = {key[len('bert.'):]: val for key, val in state_dict.items()}
                state_dict = {key: val for key, val in state_dict.items() if key in model.bert.state_dict()}
            check_diff(model.bert.state_dict(), state_dict)
            model.bert.load_state_dict(state_dict)
            logger.info('Model initialized from the pre-trained BERT weight!')

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif args.parallel or n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Data parallel!")

    if args.do_load:
        bind_model(processor, model)
        processor.load(args.load_epoch, session=args.load_dir)

    def is_freeze_param(name):
        if args.freeze_word_emb:
            if name.endswith("bert.embeddings.word_embeddings.weight"):
                logger.info(f'freezeing {name}')
                return False
        return True

    # Dump phrases
    if args.do_dump:
        if ':' not in args.predict_file:
            predict_files = [args.predict_file]
            offsets = [0]
        else:
            dirname = os.path.dirname(args.predict_file)
            basename = os.path.basename(args.predict_file)
            start, end = list(map(int, basename.split(':')))

            # skip files if possible
            if os.path.exists(args.dump_file):
                with h5py.File(args.dump_file, 'r') as f:
                    dids = list(map(int, f.keys()))
                start = int(max(dids) / 1000)
                logger.info('%s exists; starting from %d' % (args.dump_file, start))

            names = [str(i).zfill(4) for i in range(start, end)]
            predict_files = [os.path.join(dirname, name) for name in names]
            offsets = [int(each) * 1000 for each in names]

        for offset, predict_file in zip(offsets, predict_files):
            context_examples = read_squad_examples(
                context_only=True,
                input_file=predict_file, return_answers=False, draft=args.draft,
                draft_num_examples=args.draft_num_examples, append_title=args.append_title)

            for example in context_examples:
                example.doc_idx += offset

            context_features = convert_documents_to_features(
                examples=context_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride)

            logger.info("***** Running dumping on %s *****" % predict_file)
            logger.info("  Num orig examples = %d", len(context_examples))
            logger.info("  Num split examples = %d", len(context_features))
            logger.info("  Batch size = %d", args.predict_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in context_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in context_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            context_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)

            if args.local_rank == -1:
                context_sampler = SequentialSampler(context_data)
            else:
                context_sampler = DistributedSampler(context_data)
            context_dataloader = DataLoader(context_data, sampler=context_sampler,
                                            batch_size=args.predict_batch_size)

            model.eval()
            logger.info("Start dumping")

            def get_context_results():
                for (input_ids, input_mask, example_indices) in context_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    with torch.no_grad():
                        batch_start, batch_end, batch_span_logits, batch_filter_start, batch_filter_end, sp_s, sp_e = model(
                                                                                                    input_ids=input_ids,
                                                                                                    input_mask=input_mask)
                    for i, example_index in enumerate(example_indices):
                        start = batch_start[i].detach().cpu().numpy().astype(args.dtype)
                        end = batch_end[i].detach().cpu().numpy().astype(args.dtype)
                        sparse = None
                        if len(sp_s) > 0:
                            b_ssp = {ng: bb_ssp[i].detach().cpu().numpy().astype(args.dtype) for ng, bb_ssp in sp_s.items()}
                            b_esp = {ng: bb_esp[i].detach().cpu().numpy().astype(args.dtype) for ng, bb_esp in sp_e.items()}
                        span_logits = batch_span_logits[i].detach().cpu().numpy().astype(args.dtype)
                        filter_start_logits = batch_filter_start[i].detach().cpu().numpy().astype(args.dtype)
                        filter_end_logits = batch_filter_end[i].detach().cpu().numpy().astype(args.dtype)
                        context_feature = context_features[example_index.item()]
                        unique_id = int(context_feature.unique_id)
                        yield ContextResult(unique_id=unique_id,
                                            start=start,
                                            end=end,
                                            span_logits=span_logits,
                                            filter_start_logits=filter_start_logits,
                                            filter_end_logits=filter_end_logits,
                                            start_sp=b_ssp,
                                            end_sp=b_esp)

            t0 = time()
            write_hdf5(context_examples, context_features, get_context_results(),
                       args.max_answer_length, not args.do_case, args.dump_file, args.filter_threshold,
                       args.verbose_logging,
                       dense_offset=args.dense_offset, dense_scale=args.dense_scale,
                       sparse_offset=args.sparse_offset, sparse_scale=args.sparse_scale,
                       use_sparse=args.use_sparse)
            logger.info('%s: %.1f mins' % (predict_file, (time() - t0) / 60))


def bind_model(processor, model, optimizer=None):
    def save(filename, save_model=True, saver=None, **kwargs):
        if not os.path.exists(filename):
            os.makedirs(filename)
        if save_model:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            model_path = os.path.join(filename, 'model.pt')
            dummy_path = os.path.join(filename, 'dummy')
            torch.save(state, model_path)
            with open(dummy_path, 'w') as fp:
                json.dump([], fp)
            logger.info('Model saved at %s' % model_path)
        if saver is not None:
            saver(filename)

    def load(filename, load_model=True, loader=None, **kwargs):
        if load_model:
            # logger.info('%s: %s' % (filename, os.listdir(filename)))
            model_path = os.path.join(filename, 'model.pt')
            if not os.path.exists(model_path):  # for compatibility
                model_path = filename
            state = torch.load(model_path, map_location='cpu')
            try:
                model.load_state_dict(state['model'])
                if optimizer is not None:
                    optimizer.load_state_dict(state['optimizer'])
                logger.info('load okay')
            except:
                # Backward compatibility
                # model.load_state_dict(load_backward(state), strict=False)
                model.load_state_dict(state, strict=False)
                check_diff(model.state_dict(), state['model'])
            logger.info('Model loaded from %s' % model_path)
        if loader is not None:
            loader(filename)

    processor.bind(save=save, load=load)


def check_diff(model_a, model_b):
    a_set = set([a for a in model_a.keys()])
    b_set = set([b for b in model_b.keys()])
    if a_set != b_set:
        logger.info('load with different params =>')
    if len(a_set - b_set) > 0:
        logger.info('Loaded weight does not have ' + str(a_set - b_set))
    if len(b_set - a_set) > 0:
        logger.info('Model code does not have: ' + str(b_set - a_set))


def load_backward(state):
    new_state = collections.OrderedDict()
    for key, val in state.items():
        multi = False
        if key.startswith('module.'):
            multi = True
            key = key[len('module.'):]

        if key == 'true_help':
            continue
        if key.startswith('bert_q.'):
            continue
        if key.startswith('linear.'):
            continue
        if key.startswith('bert.'):
            key = 'encoder.' + key

        if multi:
            key = 'module.' + key
        new_state[key] = val
    return new_state


if __name__ == "__main__":
    main()
