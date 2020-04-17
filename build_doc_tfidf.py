#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging
import json
import copy
import pandas as pd

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

import tfidf_util
from simple_tokenizer import SimpleTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB[doc_id]


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(tfidf_util.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=tfidf_util.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([tfidf_util.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, file_path):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    doc_ids = {}
    doc_metas = {}
    nan_cnt = 0
    for filename in sorted(os.listdir(file_path)):
        print(filename)
        with open(os.path.join(file_path, filename), 'r') as f:
            articles = json.load(f)['data']
            for article in articles:
                title = article['title']
                kk = 0
                while title in doc_ids:
                    title += f'_{kk}'
                    kk += 1
                doc_ids[title] = ' '.join([par['context'] for par in article['paragraphs']])

                # Keep metadata
                doc_meta = {}
                for key, val in article.items():
                    if key != 'paragraphs':
                        doc_meta[key] = val if val == val else 'NaN'
                    else:
                        doc_meta[key] = []
                        for para in val:
                            para_meta = {}
                            for para_key, para_val in para.items():
                                if para_key != 'context':
                                    para_meta[para_key] = para_val if para_val == para_val else 'NaN'
                            doc_meta[key].append(para_meta)
                if not pd.isnull(article.get('pubmed_id', np.nan)):
                    doc_metas[str(article['pubmed_id'])] = doc_meta # For BEST (might be duplicate)
                else:
                    nan_cnt += 1
                doc_metas[article['title']] = doc_meta

    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    print('doc ids:', len(DOC2IDX))
    print('doc metas:', len(doc_metas), 'with nan', str(nan_cnt))
    # assert len(doc_ids)*2 == len(doc_metas) + nan_cnt

    # Setup worker pool
    tok_class = SimpleTokenizer
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, doc_ids)
    )
    doc_ids = list(doc_ids.keys())

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids, doc_metas)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, default=None,
                        help='Path to document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(
        args, args.file_path
    )

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.file_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=simple' %
                 (args.ngram, args.hash_size))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': 'simple',
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    tfidf_util.save_sparse_csr(filename, tfidf, metadata)
