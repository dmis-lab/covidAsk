import argparse
import math
import os
import subprocess


def run_dump_phrase(args):
    parallel = '--parallel' if args.parallel else ''
    do_case = '--do_case' if args.do_case else ''
    use_biobert = '--use_biobert' if args.use_biobert else ''
    append_title = '--append_title' if args.append_title else ''
    def get_cmd(start_doc, end_doc):
        return ["python", "run_denspi.py",
                "--metadata_dir", f"{args.metadata_dir}",
                "--data_dir", f"{args.phrase_data_dir}",
                "--predict_file", f"{start_doc}:{end_doc}",
                "--bert_model_option", f"{args.bert_model_option}",
                "--do_dump",
                "--use_sparse",
                "--filter_threshold", f"{args.filter_threshold:.2f}",
                "--dump_dir", f"{args.phrase_dump_dir}",
                "--dump_file", f"{start_doc}-{end_doc}.hdf5",
                "--max_seq_length", "512",
                "--load_dir", f"{args.load_dir}",
                "--load_epoch", f"{args.load_epoch}"] + \
                ([f"{parallel}"] if len(parallel) > 0 else []) + \
                ([f"{do_case}"] if len(do_case) > 0 else []) + \
                ([f"{use_biobert}"] if len(use_biobert) > 0 else []) + \
                ([f"{append_title}"] if len(append_title) > 0 else [])


    num_docs = args.end - args.start
    num_gpus = args.num_gpus
    num_docs_per_gpu = int(math.ceil(num_docs / num_gpus))
    start_docs = list(range(args.start, args.end, num_docs_per_gpu))
    end_docs = start_docs[1:] + [args.end]

    print(start_docs)
    print(end_docs)

    for device_idx, (start_doc, end_doc) in enumerate(zip(start_docs, end_docs)):
        print(get_cmd(start_doc, end_doc))
        subprocess.Popen(get_cmd(start_doc, end_doc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', default=None)
    parser.add_argument('--metadata_dir', default='models/bert')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--data_name', default='2020-04-10')
    parser.add_argument('--load_dir', default='models/denspi')
    parser.add_argument('--load_epoch', default='1')
    parser.add_argument('--bert_model_option', default='large_uncased')
    parser.add_argument('--append_title', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--do_case', default=False, action='store_true')
    parser.add_argument('--use_biobert', default=False, action='store_true')
    parser.add_argument('--filter_threshold', default=-1e9, type=float)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=8, type=int)
    args = parser.parse_args()

    if args.dump_dir is None:
        args.dump_dir = os.path.join('dumps_new/%s_%s' % (os.path.basename(args.load_dir),
                                                          os.path.basename(args.data_name)))
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    if not os.path.exists('logs'):
        os.makedirs('logs')

    args.phrase_data_dir = os.path.join(args.data_dir, args.data_name)
    args.phrase_dump_dir = os.path.join(args.dump_dir, 'phrase')
    if not os.path.exists(args.phrase_dump_dir):
        os.makedirs(args.phrase_dump_dir)

    return args


def main():
    args = get_args()
    run_dump_phrase(args)


if __name__ == '__main__':
    main()
