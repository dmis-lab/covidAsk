import json
import os
import argparse
import random
import h5py
import numpy as np
import scipy.sparse as sp
import logging

from tfidf_doc_ranker import TfidfDocRanker
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dump_tfidf(ranker, dumps, names, args):
    for phrase_dump, name in tqdm(zip(dumps, names)):
        with h5py.File(os.path.join(args.out_dir, name + '_tfidf.hdf5')) as f:
            for doc_id in tqdm(phrase_dump):
                if doc_id in f:
                    print('%s exists; replacing' % doc_id)
                    del f[doc_id]
                dg = f.create_group(doc_id)
                doc = phrase_dump[doc_id]
                para_lens = doc['len_per_para'][:]
                para_startend = [(sum(para_lens[:para_idx]), sum(para_lens[:para_idx+1])) for
                    para_idx in range(len(para_lens))]
                # print(doc['input_ids'].shape, doc['sparse'].shape, len(doc['word2char_start']), doc['start'].shape)
                assert doc['sparse'].shape[0] == doc['start'].shape[0], '%d vs %d'%(doc['sparse'].shape[0], doc['start'].shape[0])
                paras = [doc.attrs['context'][doc['word2char_start'][ps]:doc['word2char_end'][pe-1]] 
                         for (ps, pe) in para_startend]
                # old_paras = [k.strip() for k in doc.attrs['context'].split('[PAR]')]
                para_data = [ranker.text2spvec(para, val_idx=True) for para in paras]
                for p_idx, data in enumerate(para_data):
                    if str(p_idx) in dg:
                        print('%s exists; replacing' % str(p_idx))
                        del dg[str(p_idx)]
                    pdg = dg.create_group(str(p_idx))
                    try:
                        pdg.create_dataset('vals', data=data[0])
                        pdg.create_dataset('idxs', data=data[1])
                    except Exception as e:
                        print('Exception occured {} {}'.format(str(e), data))
                        pdg.create_dataset('vals', data=[0])
                        pdg.create_dataset('idxs', data=[0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--dump_path', default=None, type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1, type=int)
    parser.add_argument('--ranker_path', default='', type=str)
    parser.add_argument('--nfs', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isdir(args.dump_dir)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if not args.dump_path:
        dump_paths = sorted([os.path.join(args.dump_dir, name) for name in os.listdir(args.dump_dir) if 'hdf5' in name])[
                     args.start:args.end]
    else:
        dump_paths = [os.path.join(args.dump_dir, args.dump_path)]
    dump_names = [os.path.splitext(os.path.basename(path))[0] for path in dump_paths]
    print(dump_paths)
    phrase_dumps = [h5py.File(path, 'r') for path in dump_paths]

    ranker = None
    ranker = TfidfDocRanker(
        tfidf_path=args.ranker_path,
        strict=False
    )

    print('Ranker shape {} from {}'.format(ranker.doc_mat.shape, args.ranker_path))
    dump_tfidf(ranker, phrase_dumps, dump_names, args)


if __name__ == '__main__':
    main()
