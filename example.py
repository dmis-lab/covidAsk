import argparse
import requests
from covidask import covidAsk


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Server ports
    parser.add_argument('--base_ip', default='http://localhost', type=str)
    parser.add_argument('--index_port', default='9030', type=str)

    # Search Hyperparameters
    parser.add_argument('--max_answer_length', default=20, type=int)
    parser.add_argument('--start_top_k', default=1000, type=int)
    parser.add_argument('--mid_top_k', default=100, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--doc_top_k', default=5, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--sparse_weight', default=0.05, type=float)
    parser.add_argument('--search_strategy', default='hybrid')
    args = parser.parse_args()

    covidask = covidAsk(base_ip=args.base_ip, index_port=args.index_port, args=args)

    query = 'Is there concrete evidence for the presence of asymptomatic transmissions?'
    result = covidask.query(query)
    print(f'\nTop {args.top_k} answers to a question: {query}')
    for r_i, r in enumerate(result["ret"]):
        print('[Answer ' + str(r_i+1) + ']: ' + r["answer"])
