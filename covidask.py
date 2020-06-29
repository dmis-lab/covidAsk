import json
import argparse
import torch
import tokenization
import os
import random
import numpy as np
import requests
import logging
import math
import ssl
import best
import copy
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from time import time
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from requests_futures.sessions import FuturesSession
from tqdm import tqdm
from collections import namedtuple
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from transformers import AutoConfig

from serve_utils import load_caches, parse_example, get_cached, get_search
from modeling import DenSPI
from tfidf_doc_ranker import TfidfDocRanker
from run_denspi import check_diff
from pre import SquadExample, convert_questions_to_features
from post import convert_question_features_to_dataloader, get_question_results
from mips_phrase import MIPS
from eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, drqa_regex_match_score,\
                       drqa_metric_max_over_ground_truths, drqa_normalize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class covidAsk(object):
    def __init__(self, base_ip='http://localhost', query_port='-1', doc_port='-1', index_port='-1', args=None):
        self.args = args

        # IP and Ports
        self.base_ip = base_ip
        self.query_port = query_port
        self.doc_port = doc_port
        self.index_port = index_port
        logger.info(f'Query address: {self.get_address(self.query_port)}')
        logger.info(f'Doc address: {self.get_address(self.doc_port)}')
        logger.info(f'Index address: {self.get_address(self.index_port)}')

        # Saved objects
        self.mips = None

    def load_query_encoder(self, device, args):
        # Configure paths for query encoder serving
        vocab_path = os.path.join(args.metadata_dir, args.vocab_name)

        # Load pretrained QueryEncoder
        bert_config = AutoConfig.from_pretrained(
            'bert-base-uncased' if not (args.bert_model_option == 'large_uncased') else 'bert-large-uncased',
            cache_dir='cache',
        )
        model = DenSPI.from_pretrained(
            args.query_encoder_path,
            config=bert_config,
        )
        logger.info('Model loaded from %s' % args.query_encoder_path)
        model.to(device)

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=not args.do_case)
        logger.info('Model loaded from %s' % args.query_encoder_path)
        logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
        return model, tokenizer

    def get_question_dataloader(self, questions, tokenizer, batch_size):
        question_examples = [SquadExample(qas_id='qs', question_text=q) for q in questions]
        query_features = convert_questions_to_features(
            examples=question_examples,
            tokenizer=tokenizer,
            max_query_length=64
        )
        question_dataloader = convert_question_features_to_dataloader(
            query_features,
            fp16=False, local_rank=-1,
            predict_batch_size=batch_size
        )
        return question_dataloader, question_examples, query_features

    def serve_query_encoder(self, query_port, args):
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = self.load_query_encoder(device, args)

        # Define query to vector function
        def query2vec(queries):
            # queries = [query[:-1] if query.endswith('?') else query for query in queries]
            question_dataloader, question_examples, query_features = self.get_question_dataloader(
                queries, tokenizer, batch_size=24
            )
            query_encoder.eval()
            question_results = get_question_results(
                question_examples, query_features, question_dataloader, device, query_encoder
            )
            outs = []
            for qr_idx, question_result in enumerate(question_results):
                for ngram in question_result.sparse.keys():
                    question_result.sparse[ngram] = question_result.sparse[ngram].tolist()
                out = (
                    question_result.start.tolist(), question_result.end.tolist(),
                    question_result.sparse, question_result.input_ids
                )
                outs.append(out)
            return outs

        # Serve query encoder
        app = Flask(__name__)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/batch_api', methods=['POST'])
        def batch_api():
            batch_query = json.loads(request.form['query'])
            outs = query2vec(batch_query)
            return jsonify(outs)

        logger.info(f'Starting QueryEncoder server at {self.get_address(query_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(query_port)
        IOLoop.instance().start()

    def load_phrase_index(self, args, dump_only=False):
        if self.mips is not None:
            return self.mips

        # Configure paths for index serving
        phrase_dump_dir = os.path.join(args.dump_dir, args.phrase_dir)
        tfidf_dump_dir = os.path.join(args.dump_dir, args.tfidf_dir)
        index_dir = os.path.join(args.dump_dir, args.index_dir)
        index_path = os.path.join(index_dir, args.index_name)
        idx2id_path = os.path.join(index_dir, args.idx2id_name)
        max_norm_path = os.path.join(index_dir, 'max_norm.json')

        # Load mips
        mips_init = MIPS
        mips = mips_init(
            phrase_dump_dir=phrase_dump_dir,
            tfidf_dump_dir=tfidf_dump_dir,
            start_index_path=index_path,
            idx2id_path=idx2id_path,
            max_norm_path=max_norm_path,
            doc_rank_fn={
                'doc_scores': self.get_doc_scores, 'top_docs': self.get_top_docs, 'doc_meta': self.get_doc_meta,
                'spvec': self.get_q_spvecs
            },
            cuda=args.cuda, dump_only=dump_only
        )
        return mips

    def best_search(self, query, kcw_path=None):
        t0 = time()
        # Type filter
        ent_types = [
            "gene", "drug", "chemical compound", "target", "disease",
            "toxin", "transcription factor", "mirna", "pathway", "mutation"
        ]
        query_type = "All Entity Type"
        for ent_type in ent_types:
            if ent_type in query.lower():
                query_type = ent_type
                break

        # Stopwords and filtering for BEST queries
        if not os.path.exists(os.path.join(os.path.expanduser('~'), 'nltk_data/corpora/stopwords')):
            nltk.download('stopwords')
        if not os.path.exists(os.path.join(os.path.expanduser('~'), 'nltk_data/tokenizers/punkt')):
            nltk.download('punkt')
        stop_words = set(stopwords.words('english') + ['?'] + ['Why', 'What', 'How', 'Where', 'When', 'Who'])
        entity_set = [
            'COVID-19', 'SARS-CoV-2', 'hypertension', 'diabetes', 'heart', 'disease', 'obese', 'death',
            'HCoV-19', 'HCoV', 'coronavirus', 'symptoms', 'incubation', 'periods', 'period', 'quarantine',
            'asymptomatic', 'transmissions', 'fecal', 'excretion', 'decline', 'Wuhan', 'mortality',
            'patients', 'stay', 'reproduction', 'risk', 'factor', 'factors', 'pregnancy', 'interval', 'absent',
            'reported', 'length', 'diagnosed', 'United', 'States', 'isolated', 'CDC', 'WHO', 'vaccine',
            'negative', 'animals', 'airbone', 'spread', 'blood', 'sanitizer', 'controlled', 'illness', 'friends',
        ]
        query_tokens = word_tokenize(query)
        new_query = ''
        for idx, query_token in enumerate(query_tokens):
            if query_token not in stop_words:
                if query_token in [e.lower() for e in entity_set]:
                    new_query += query_token + ' '
        if new_query == '':
            new_query = 'COVID-19'

        # Get BEST result
        q = best.BESTQuery(new_query, noAbsTxt=False, filterObjectName=query_type)
        r = best.getRelevantBioEntities(q)

        # No result
        if len(r) == 1 and r[0]['rank'] == 0 and len(r[0].keys()) == 1:
            t1 = time()
            return {'ret': [], 'time': int(1000 * (t1 - t0))}

        parsed_result = {
            'context': '',
            'title': '',
            'doc_idx': None,
            'start_idx': 0,
            'end_idx': 0,
            'score': 0,
            'metadata': {
                'pubmed_id': ''
            },
            'answer': ''
        }
        outs = []
        metas = self.get_doc_meta([r_['PMIDs'][0] for r_ in r])
        for r_idx, (r_, meta) in enumerate(zip(r, metas)):
            parsed_result['context'] = r_['abstracts'][0]
            parsed_result['score'] = r_['score']
            parsed_result['answer'] = r_['entityName']
            parsed_result['metadata'] = meta
            if len(parsed_result['metadata']) == 0:
                parsed_result['metadata']['pubmed_id'] = int(r_['PMIDs'][0])
            outs.append(copy.deepcopy(parsed_result))

        t1 = time()
        return {'ret': outs, 'time': int(1000 * (t1 - t0))}

    def serve_phrase_index(self, index_port, args):
        if index_port == '80':
            app = Flask(__name__, static_url_path='/static', static_folder="static",
                template_folder="templates")
            app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
            CORS(app)
            @app.before_request
            def before_request():
                if request.url.startswith('http://'):
                    # url = request.url.replace('http://', 'https://', 1)
                    code = 301
                    return redirect('https://covidask.korea.ac.kr', code=code)
            http_server = HTTPServer(WSGIContainer(app))
            http_server.listen(index_port)
            IOLoop.instance().start()
            return

        dev_str = '_dev' if args.develop else ''
        args.examples_path = os.path.join(f'static{dev_str}', args.examples_path)
        args.top10_examples_path = os.path.join(f'static{dev_str}', args.top10_examples_path)

        # Load mips
        self.mips = self.load_phrase_index(args)
        app = Flask(__name__, static_url_path='/static' + dev_str, static_folder="static" + dev_str,
            template_folder="templates" + dev_str)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        # From serve_utils
        cached_set = load_caches(args)
        index_example_set, search_examples, inverted_examples, query_entity_ids = parse_example(args)

        def batch_search(batch_query, max_answer_length=20, start_top_k=1000, mid_top_k=100, top_k=10, doc_top_k=5,
                         nprobe=64, sparse_weight=0.05, search_strategy='dense_first', aggregate=False, meta_scale=10):
            t0 = time()
            outs, _ = self.embed_query(batch_query)()
            start = np.concatenate([out[0] for out in outs], 0)
            end = np.concatenate([out[1] for out in outs], 0)
            sparse_uni = [out[2]['1'][1:len(out[3])+1] for out in outs]
            sparse_bi = [out[2]['2'][1:len(out[3])+1] for out in outs]
            input_ids = [out[3] for out in outs]
            query_vec = np.concatenate([start, end, [[1]]*len(outs)], 1)

            rets = self.mips.search(
                query_vec, (input_ids, sparse_uni, sparse_bi), q_texts=batch_query, nprobe=nprobe,
                doc_top_k=doc_top_k, start_top_k=start_top_k, mid_top_k=mid_top_k, top_k=top_k,
                search_strategy=search_strategy, filter_=args.filter, max_answer_length=max_answer_length,
                sparse_weight=sparse_weight, aggregate=aggregate, meta_scale=meta_scale
            )
            t1 = time()
            out = {'ret': rets, 'time': int(1000 * (t1 - t0))}
            return out


        @app.route('/')
        def index():
            return render_template(f'index.html')

        @app.route('/files/<path:path>')
        def static_files(path):
            return app.send_static_file('files/' + path)

        @app.route('/cached_example', methods=['GET'])
        def cached_example():
            start_time = time()
            q_id = request.args['q_id']
            res, query, query_info = get_cached(search_examples, q_id, query_entity_ids, cached_set)
            latency = time() - start_time
            latency = format(latency, ".3f")
            return render_template(f'cached.html', latency=latency, res=res, query=query, query_info=query_info)

        @app.route('/search', methods=['GET'])
        def search():
            query = request.args['query']
            params = {
                'strat': request.args['strat'] if 'strat' in request.args else 'dense_first',
                'm_a_l': (int(request.args['max_answer_length']) if 'max_answer_length' in request.args
                          else int(args.max_answer_length)),
                't_k': int(request.args['top_k']) if 'top_k' in request.args else int(args.top_k),
                'n_p': int(request.args['nprobe']) if 'nprobe' in request.args else int(args.nprobe),
                'd_t_k': int(request.args['doc_top_k']) if 'doc_top_k' in request.args else int(args.doc_top_k),
                's_w': (float(request.args['sparse_weight']) if 'sparse_weight' in request.args
                        else float(args.sparse_weight)),
                'a_g': (request.args['aggregate'] == 'True') if 'aggregate' in request.args else True,
                'm_s': float(request.args['meta_scale']) if 'meta_scale' in request.args else 100

            }
            logger.info(f'{params["strat"]} search strategy is used.')

            out = batch_search(
                [query],
                max_answer_length = params['m_a_l'],
                top_k = params['t_k'],
                nprobe = params['n_p'],
                search_strategy=params['strat'], # [DFS, SFS, Hybrid]
                doc_top_k = params['d_t_k'],
                sparse_weight = params['s_w'],
                aggregate = params['a_g'],
                meta_scale = params['m_s']
            )
            out['ret'] = out['ret'][0]
            # out['ret'] = out['ret'][:3] # Get top 3 only
            b_out = self.best_search(query, kcw_path=args.examples_path)

            res, query, query_info = get_search(inverted_examples, search_examples, query_entity_ids, query, out, b_out)
            return render_template(f'search.html', latency=out['time'],
                    res=res, query=query, query_info=query_info, params=params)

        # This one uses a default hyperparameters
        @app.route('/api', methods=['GET'])
        def api():
            query = request.args['query']
            strat = request.args['strat'] if 'strat' in request.args else 'dense_first'
            out = batch_search(
                [query],
                max_answer_length=args.max_answer_length,
                top_k=args.top_k,
                nprobe=args.nprobe,
                search_strategy=strat,
                doc_top_k=args.doc_top_k
            )
            out['ret'] = out['ret'][0]
            return jsonify(out)

        @app.route('/batch_api', methods=['POST'])
        def batch_api():
            batch_query = json.loads(request.form['query'])
            max_answer_length = int(request.form['max_answer_length'])
            start_top_k = int(request.form['start_top_k'])
            mid_top_k = int(request.form['mid_top_k'])
            top_k = int(request.form['top_k'])
            doc_top_k = int(request.form['doc_top_k'])
            nprobe = int(request.form['nprobe'])
            sparse_weight = float(request.form['sparse_weight'])
            meta_scale = float(request.form['meta_scale'])
            strat = request.form['strat']
            out = batch_search(
                batch_query,
                max_answer_length=max_answer_length,
                start_top_k=start_top_k,
                mid_top_k=mid_top_k,
                top_k=top_k,
                doc_top_k=doc_top_k,
                nprobe=nprobe,
                sparse_weight=sparse_weight,
                search_strategy=strat,
                aggregate=args.aggregate,
                meta_scale=meta_scale
            )
            return jsonify(out)

        @app.route('/get_examples', methods=['GET'])
        def get_examples():
            return render_template(f'example.html', res = index_example_set)

        @app.route('/set_query_port', methods=['GET'])
        def set_query_port():
            self.query_port = request.args['query_port']
            return jsonify(f'Query port set to {self.query_port}')

        if self.query_port is None:
            logger.info('You must set self.query_port for querying. You can use self.update_query_port() later on.')
        logger.info(f'Starting Index server at {self.get_address(index_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(index_port)
        IOLoop.instance().start()

    def serve_doc_ranker(self, doc_port, args):
        doc_ranker_path = os.path.join(args.dump_dir, args.doc_ranker_name)
        doc_ranker = TfidfDocRanker(doc_ranker_path, strict=False)
        app = Flask(__name__)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/doc_scores', methods=['POST'])
        def doc_scores():
            batch_query = json.loads(request.form['query'])
            meta_scale = float(request.form['meta_scale'])
            doc_idxs = json.loads(request.form['doc_idxs'])
            outs = doc_ranker.batch_doc_scores(batch_query, doc_idxs, meta_scale=meta_scale)
            logger.info(f'Returning {len(outs)} from batch_doc_scores')
            return jsonify(outs)

        @app.route('/top_docs', methods=['POST'])
        def top_docs():
            batch_query = json.loads(request.form['query'])
            top_k = int(request.form['top_k'])
            meta_scale = float(request.form['meta_scale'])
            batch_results = doc_ranker.batch_closest_docs(batch_query, meta_scale=meta_scale, k=top_k)
            top_idxs = [b[0] for b in batch_results]
            top_scores = [b[1].tolist() for b in batch_results]
            logger.info(f'Returning from batch_closest_docs')
            return jsonify([top_idxs, top_scores])

        @app.route('/doc_meta', methods=['POST'])
        def doc_meta():
            batch_pmid = json.loads(request.form['pmid'])
            doc_meta = doc_ranker.batch_doc_meta(batch_pmid)
            # logger.info(f'Returning {len(doc_meta)} metadata from batch_doc_meta')
            return jsonify(doc_meta)

        @app.route('/text2spvec', methods=['POST'])
        def text2spvec():
            batch_query = json.loads(request.form['query'])
            q_spvecs = [doc_ranker.text2spvec(q, val_idx=True) for q in batch_query]
            q_vals = [q_spvec[0].tolist() for q_spvec in q_spvecs]
            q_idxs = [q_spvec[1].tolist() for q_spvec in q_spvecs]
            logger.info(f'Returning {len(q_vals), len(q_idxs)} q_spvecs')
            return jsonify([q_vals, q_idxs])

        logger.info(f'Starting DocRanker server at {self.get_address(doc_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(doc_port)
        IOLoop.instance().start()

    def get_address(self, port):
        assert self.base_ip is not None
        if len(port) != 0:
            return self.base_ip + ':' + port
        else:
            return self.base_ip

    def embed_query(self, batch_query):
        emb_session = FuturesSession()
        r = emb_session.post(self.get_address(self.query_port) + '/batch_api', data={'query': json.dumps(batch_query)})
        def map_():
            result = r.result()
            emb = result.json()
            return emb, result.elapsed.total_seconds() * 1000
        return map_

    def embed_all_query(self, questions, batch_size=16):
        all_outs = []
        for q_idx in tqdm(range(0, len(questions), batch_size)):
            outs, _ = self.embed_query(questions[q_idx:q_idx+batch_size])()
            all_outs += outs
        start = np.concatenate([out[0] for out in all_outs], 0)
        end = np.concatenate([out[1] for out in all_outs], 0)

        # input ids are truncated (no [CLS], [SEP]) but sparse vals are not ([CLS] max_len [SEP])
        sparse_uni = [out[2]['1'][1:len(out[3])+1] for out in all_outs]
        sparse_bi = [out[2]['2'][1:len(out[3])+1] for out in all_outs]
        input_ids = [out[3] for out in all_outs]
        query_vec = np.concatenate([start, end, [[1]]*len(all_outs)], 1)
        logger.info(f'Query reps: {query_vec.shape}, {len(input_ids)}, {len(sparse_uni)}, {len(sparse_bi)}')
        return query_vec, input_ids, sparse_uni, sparse_bi

    def query(self, query, search_strategy='dense_first'):
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        params = {'query': query, 'strat': search_strategy}
        res = requests.get(self.get_address(self.index_port) + '/api', params=params, verify=False)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {query}')
            logger.info(res.text)
        return outs

    def batch_query(self, batch_query, max_answer_length=20, start_top_k=1000, mid_top_k=100, top_k=10, doc_top_k=5,
                    nprobe=64, sparse_weight=0.05, search_strategy='dense_first', meta_scale=10):
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        post_data = {
            'query': json.dumps(batch_query),
            'max_answer_length': max_answer_length,
            'start_top_k': start_top_k,
            'mid_top_k': mid_top_k,
            'top_k': top_k,
            'doc_top_k': doc_top_k,
            'nprobe': nprobe,
            'sparse_weight': sparse_weight,
            'strat': search_strategy,
            'meta_scale': meta_scale
        }
        res = requests.post(self.get_address(self.index_port) + '/batch_api', data=post_data, verify=False)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {batch_query}')
            logger.info(res.text)
        return outs

    def get_doc_scores(self, batch_query, doc_idxs, meta_scale=10):
        post_data = {
            'query': json.dumps(batch_query),
            'meta_scale': meta_scale,
            'doc_idxs': json.dumps(doc_idxs)
        }
        res = requests.post(self.get_address(self.doc_port) + '/doc_scores', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for {doc_idxs}')
            logger.info(res.text)
        return result

    def get_top_docs(self, batch_query, top_k, meta_scale=10):
        post_data = {
            'query': json.dumps(batch_query),
            'meta_scale': meta_scale,
            'top_k': top_k
        }
        res = requests.post(self.get_address(self.doc_port) + '/top_docs', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for {top_k}')
            logger.info(res.text)
        return result

    def get_doc_meta(self, batch_pmid):
        post_data = {
            'pmid': json.dumps(batch_pmid)
        }
        res = requests.post(self.get_address(self.doc_port) + '/doc_meta', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for {pmid}')
            logger.info(res.text)
        return result

    def get_q_spvecs(self, batch_query):
        post_data = {'query': json.dumps(batch_query)}
        res = requests.post(self.get_address(self.doc_port) + '/text2spvec', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            result = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {batch_query}')
            logger.info(res.text)
        return result

    def update_query_port(self, query_port):
        params = {'query_port': query_port}
        res = requests.get(self.get_address(self.index_port) + '/set_query_port', params=params)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for port {query_port}')
            logger.info(res.text)
        logger.info(outs)

    def load_qa_pairs(self, data_path, args):
        q_ids = []
        questions = []
        answers = []
        data = json.load(open(data_path))['data']
        for item in data:
            q_id = item['id']
            question = item['question'] # + '?' # For NQ
            answer = item['answers']
            q_ids.append(q_id)
            questions.append(question)
            answers.append(answer)
        questions = [query.replace('[MASK] .', '_') for query in questions] # For RE datasets (no diff)

        if args.draft:
            rand_idxs = np.random.choice(len(questions), 20, replace=False)
            q_ids = np.array(q_ids)[rand_idxs].tolist()
            questions = np.array(questions)[rand_idxs].tolist()
            answers = np.array(answers)[rand_idxs].tolist()
        logger.info(f'Sample Q ({q_ids[0]}): {questions[0]}, A: {answers[0]}')
        logger.info(f'Evaluating {len(questions)} questions from {args.test_path}')
        return q_ids, questions, answers

    def eval_inmemory(self, args):
        # Load dataset and encode queries
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)

        # Search
        step = args.eval_batch_size
        predictions = []
        for q_idx in tqdm(range(0, len(questions), step)):
            result = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, max_answer_length=args.max_answer_length,
                sparse_weight=args.sparse_weight, aggregate=args.aggregate, meta_scale=args.meta_scale
            )
            prediction = [[ret['answer'] for ret in out] for out in result]
            predictions += prediction

        self.evaluate_results(predictions, qids, questions, answers, args)

    def eval_request(self, args):
        # Load dataset
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)

        # Run batch_query and evaluate
        step = args.eval_batch_size
        predictions = []
        evidences = []
        for q_idx in tqdm(range(0, len(questions), step)):
            result = self.batch_query(
                questions[q_idx:q_idx+step],
                max_answer_length=args.max_answer_length,
                start_top_k=args.start_top_k,
                mid_top_k=args.mid_top_k,
                top_k=args.top_k,
                doc_top_k=args.doc_top_k,
                nprobe=args.nprobe,
                sparse_weight=args.sparse_weight,
                search_strategy=args.search_strategy,
                meta_scale=args.meta_scale
            )
            prediction = [[ret['answer'] for ret in out] for out in result['ret']]
            evidence = [[ret['context'][ret['sent_start']:ret['sent_end']] for ret in out] for out in result['ret']]
            predictions += prediction
            evidences += evidence
        self.evaluate_results(predictions, qids, questions, answers, args, evidences=evidences)

    def evaluate_results(self, predictions, qids, questions, answers, args, evidences=None):
        # Filter if there's candidate
        if args.candidate_path is not None:
            candidates = set()
            with open(args.candidate_path) as f:
                for line in f:
                    line = line.strip().lower()
                    candidates.add(line)
            logger.info(f'{len(candidates)} candidates are loaded from {args.candidate_path}')
            topk_preds = [list(filter(lambda x: (x in candidates) or (x.lower() in candidates), a)) for a in predictions]
            topk_preds = [a if len(a) > 0 else [''] for a in topk_preds]
            predictions = topk_preds[:]
            top1_preds = [a[0] for a in topk_preds]
        else:
            predictions = [a if len(a) > 0 else [''] for a in predictions]
            top1_preds = [a[0] for a in predictions]
        no_ans = sum([a == '' for a in top1_preds])
        logger.info(f'no_ans/all: {no_ans}, {len(top1_preds)}')
        logger.info(f'Evaluating {len(top1_preds)} answers.')

        # Get em/f1
        f1s, ems = [], []
        for prediction, groundtruth in zip(top1_preds, answers):
            if len(groundtruth)==0:
                f1s.append(0)
                ems.append(0)
                continue
            f1s.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
            ems.append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
        final_f1, final_em = np.mean(f1s), np.mean(ems)
        logger.info('EM: %.2f, F1: %.2f'%(final_em * 100, final_f1 * 100))

        # Top 1/k em (or regex em)
        exact_match_topk = 0
        exact_match_top1 = 0
        f1_score_topk = 0
        f1_score_top1 = 0
        pred_out = {}
        for i in range(len(predictions)):
            # For debugging
            if i < 3:
                logger.info(f'{i+1}) {questions[i]}')
                logger.info(f'=> groudtruths: {answers[i]}, prediction: {predictions[i][:5]}')

            match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
            em_topk = max([drqa_metric_max_over_ground_truths(
                match_fn, prediction, answers[i]
            ) for prediction in predictions[i]])
            em_top1 = drqa_metric_max_over_ground_truths(
                match_fn, top1_preds[i], answers[i]
            )
            exact_match_topk += em_topk
            exact_match_top1 += em_top1

            f1_topk = 0
            f1_top1 = 0
            if not args.regex:
                match_fn = lambda x, y: f1_score(x, y)[0]
                f1_topk = max([drqa_metric_max_over_ground_truths(
                    match_fn, prediction, answers[i]
                ) for prediction in predictions[i]])
                f1_top1 = drqa_metric_max_over_ground_truths(
                    match_fn, top1_preds[i], answers[i]
                )
                f1_score_topk += f1_topk
                f1_score_top1 += f1_top1

            pred_out[qids[i]] = {
                    'question': questions[i],
                    'answer': answers[i], 'prediction': predictions[i],
                    'evidence': evidences[i] if evidences is not None else '',
                    'em_top1': bool(em_top1), f'em_top{args.top_k}': bool(em_topk),
                    'f1_top1': f1_top1, f'f1_top{args.top_k}': f1_topk
            }
        total = len(predictions)
        exact_match_top1 = 100.0 * exact_match_top1 / total
        f1_score_top1 = 100.0 * f1_score_top1 / total
        logger.info({'exact_match_top1': exact_match_top1, 'f1_score_top1': f1_score_top1})
        exact_match_topk = 100.0 * exact_match_topk / total
        f1_score_topk = 100.0 * f1_score_topk / total
        logger.info({f'exact_match_top{args.top_k}': exact_match_topk, f'f1_score_top{args.top_k}': f1_score_topk})

        # Dump predictions
        if not os.path.exists('pred'):
            os.makedirs('pred')
        pred_path = os.path.join('pred', os.path.splitext(os.path.basename(args.test_path))[0] + '.pred')
        logger.info(f'Saving prediction file to {pred_path}')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f)

    def eval_sent(self, args):
        # Load dataset
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)

        # Run batch_query and evaluate
        step = args.eval_batch_size
        predictions = []
        evidences = []
        for q_idx in tqdm(range(0, len(questions), step)):
            result = self.batch_query(
                questions[q_idx:q_idx+step],
                max_answer_length=args.max_answer_length,
                start_top_k=args.start_top_k,
                mid_top_k=args.mid_top_k,
                top_k=args.top_k,
                doc_top_k=args.doc_top_k,
                nprobe=args.nprobe,
                sparse_weight=args.sparse_weight,
                search_strategy=args.search_strategy,
                meta_scale=args.meta_scale
            )
            prediction = [[ret['context'][ret['sent_start']:ret['sent_end']] for ret in out] for out in result['ret']]
            predictions += prediction

        return self.evaluate_sent_results(predictions, qids, questions, answers, args)

    def eval_sent_inm(self, args):
        # Load dataset and encode queries
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)

        # Search
        step = args.eval_batch_size
        predictions = []
        for q_idx in tqdm(range(0, len(questions), step)):
            result = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, max_answer_length=args.max_answer_length,
                sparse_weight=args.sparse_weight, aggregate=args.aggregate, meta_scale=args.meta_scale
            )
            prediction = [[ret['context'][ret['sent_start']:ret['sent_end']] for ret in out] for out in result]
            predictions += prediction

        return self.evaluate_sent_results(predictions, qids, questions, answers, args)

    def evaluate_sent_results(self, predictions, qids, questions, answers, args):
        _start_time = time()

        # Evaluate based on recall in sentence
        recall_1 = 0.0
        recall_k = 0.0
        mrr_k = 0.00
        prec_k = 0.00
        pred_out = {}
        assert len(predictions) == len(answers)
        for i, (topk_pred, answer) in enumerate(zip(predictions, answers)):
            single_recall_1 = 0.0
            single_recall_k = 0.0
            single_mrr_k = 0.0
            single_prec_k = 0.0
            for rank, sent in enumerate(topk_pred):
                if any([cand.lower() in sent.lower() for cand in answer]):
                    matched = np.array(answer)[[aa for aa, cand in enumerate(answer) if cand.lower() in sent.lower()]]
                    predictions[i][rank] = predictions[i][rank] + f' => matched with {matched}'
                    single_recall_k = 1.0
                    if rank == 0:
                        single_recall_1 = 1.0

                    if rank < 50:
                        single_prec_k += 1.0

                # First MRR occurrence
                if single_mrr_k == 0 and single_recall_k != 0:
                    single_mrr_k = 1.0 / (rank+1)
            single_prec_k = single_prec_k / 50
            # single_prec_k = single_prec_k / args.top_k

            recall_1 += single_recall_1
            recall_k += single_recall_k
            mrr_k += single_mrr_k
            prec_k += single_prec_k

            pred_out[qids[i]] = {
                'question': questions[i],
                'answer': answers[i],
                'prediction': predictions[i],
                'recall_1': bool(single_recall_1),
                'recall_k': bool(single_recall_k),
                'prec_k': f'{single_prec_k:.3f}',
                'mrr_k': f'{single_mrr_k:.3f}',
            }

        recall_1 = recall_1 / len(predictions)
        recall_k = recall_k / len(predictions)
        mrr_k = mrr_k / len(predictions)
        prec_k = prec_k / len(predictions)
        logger.info(f'Recall@1: {recall_1:.4f}')
        logger.info(f'Recall@{args.top_k}: {recall_k:.4f}')
        logger.info(f'Precision@{args.top_k}: {prec_k:.4f}')
        logger.info(f'MRR@{args.top_k}: {mrr_k:.4f}')

        # Dump predictions
        if not os.path.exists('pred'):
            os.makedirs('pred')
        pred_path = os.path.join('pred', os.path.splitext(os.path.basename(args.test_path))[0] + '.pred')
        logger.info(f'Saving prediction file to {pred_path}')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f)

        _elapsed_time = time() - _start_time
        scores = {'recall_1': recall_1, 'recall_k': recall_k, 'mrr_k': mrr_k, 'time': _elapsed_time}
        return scores

    def save_top_k(self, args):
        # Load dataset and encode queries
        qids, questions, answers = self.load_qa_pairs(args.test_path, args)
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)
        args.examples_path = os.path.join('static', args.examples_path)

        # Search
        step = args.eval_batch_size
        predictions = []
        b_out = []
        for q_idx in tqdm(range(0, len(questions), step)):
            prediction = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, max_answer_length=args.max_answer_length,
                sparse_weight=args.sparse_weight, aggregate=args.aggregate, meta_scale=args.meta_scale
            )
            predictions += prediction
            b_out += [self.best_search(query, args.examples_path) for query in questions[q_idx:q_idx+step]]

        self.evaluate_results(predictions, qids, questions, answers, args)

        # Dump predictions
        if not os.path.exists('pred'):
            os.makedirs('pred')
        with open(os.path.join('pred', f'top{args.top_k}_{os.path.basename(args.test_path)}'), 'w') as f:
            json.dump({'data': {q: {'denspi': p, 'best': b} for q, p, b in zip(qids, predictions, b_out)}}, f, indent=2)
        logger.info('Saving top k done.')

    def eval_trec(self, args):
        questions = []
        qids = []

        # Get questions from xml
        import xml.etree.ElementTree as ET
        tree = ET.parse(args.test_path)
        for elem in tree.iter():
            if elem.tag == 'topic':
                questions.append(elem.find('question').text)
                # questions.append(elem.find('query').text)
                qids.append(elem.attrib['number'])
        query_vec, input_ids, sparse_uni, sparse_bi = self.embed_all_query(questions)

        # Load MIPS
        self.mips = self.load_phrase_index(args)
        args.examples_path = os.path.join('static', args.examples_path)

        # Search
        step = args.eval_batch_size
        predictions = []
        pred_answers = []
        b_out = []
        for q_idx in tqdm(range(0, len(questions), step)):
            prediction = self.mips.search(
                query_vec[q_idx:q_idx+step],
                (input_ids[q_idx:q_idx+step], sparse_uni[q_idx:q_idx+step], sparse_bi[q_idx:q_idx+step]),
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                doc_top_k=args.doc_top_k, start_top_k=args.start_top_k, mid_top_k=args.mid_top_k, top_k=args.top_k,
                search_strategy=args.search_strategy, filter_=args.filter, max_answer_length=args.max_answer_length,
                sparse_weight=args.sparse_weight, aggregate=args.aggregate, meta_scale=args.meta_scale
            )
            pred_answers += [[ret['answer'] for ret in out] for out in prediction]
            predictions += prediction

        # Dump predictions
        if not os.path.exists('pred'):
            os.makedirs('pred')
        out_file = open(os.path.join('pred', os.path.basename(args.test_path) + '.trec'), 'w')
        for qid, question, prediction in zip(qids, questions, predictions):
            for rank, pred in enumerate(prediction):
                out_file.write(f'{qid} Q0 {pred["metadata"]["cord_uid"]} {rank} {pred["score"]} dmis-r1-t1\n')
        out_file.close()
        logger.info(f'TREC-COVID result saved in {out_file.name}')

        # Checking errors
        import subprocess
        command = f"perl check_sub.pl {out_file.name}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        command = f"cat {os.path.basename(out_file.name)}.errlog"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)

        # Pred log file
        self.evaluate_results(pred_answers, qids, questions, [['']]*len(questions), args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser.add_argument('--metadata_dir', default='models/bert', type=str)
    parser.add_argument("--vocab_name", default='vocab.txt', type=str)
    parser.add_argument("--bert_config_name", default='bert_config.json', type=str)
    parser.add_argument("--bert_model_option", default='large_uncased', type=str)
    parser.add_argument("--do_case", default=False, action='store_true')
    parser.add_argument("--use_biobert", default=False, action='store_true')
    parser.add_argument("--query_encoder_path", default='models/denspi/1/model.pt', type=str)
    parser.add_argument("--query_port", default='-1', type=str)

    # DocRanker
    parser.add_argument('--doc_ranker_name', default='2020-04-10-recent-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
    parser.add_argument('--doc_port', default='-1', type=str)

    # PhraseIndex
    parser.add_argument('--dump_dir', default='dumps/denspi_2020-04-10')
    parser.add_argument('--phrase_dir', default='phrase')
    parser.add_argument('--tfidf_dir', default='tfidf')
    parser.add_argument('--index_dir', default='16384_hnsw_SQ8')
    parser.add_argument('--index_name', default='index.faiss')
    parser.add_argument('--idx2id_name', default='idx2id.hdf5')
    parser.add_argument('--index_port', default='-1', type=str)

    # These can be dynamically changed.
    parser.add_argument('--max_answer_length', default=20, type=int)
    parser.add_argument('--start_top_k', default=1000, type=int)
    parser.add_argument('--mid_top_k', default=100, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--doc_top_k', default=5, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--sparse_weight', default=0.05, type=float)
    parser.add_argument('--meta_scale', default=10, type=float)
    parser.add_argument('--search_strategy', default='dense_first')
    parser.add_argument('--aggregate', default=False, action='store_true')
    parser.add_argument('--filter', default=False, action='store_true')
    parser.add_argument('--no_para', default=False, action='store_true')

    # Serving options
    parser.add_argument('--examples_path', default='queries/examples.json')
    parser.add_argument('--top10_examples_path', default='queries/top10_preds.json')
    parser.add_argument('--develop', default=False, action='store_true')

    # Evaluation
    parser.add_argument('--test_path', default='data/eval/kaggle_cdc_who_combined.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=10, type=int)
    parser.add_argument('--top_phrase_path', default='top_phrases.json')

    # Run mode
    parser.add_argument('--base_ip', default='http://163.152.20.133')
    parser.add_argument('--run_mode', default='batch_query')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    covidask = covidAsk(
        base_ip=args.base_ip,
        query_port=args.query_port,
        doc_port=args.doc_port,
        index_port=args.index_port,
        args=args
    )

    # Usages
    if args.run_mode == 'q_serve':
        covidask.serve_query_encoder(args.query_port, args)

    elif args.run_mode == 'd_serve':
        covidask.serve_doc_ranker(args.doc_port, args)

    elif args.run_mode == 'p_serve':
        covidask.serve_phrase_index(args.index_port, args)

    elif args.run_mode == 'query':
        query = 'Which Lisp framework has been developed for image processing?'
        # query = ' Several genetic factors have been related to HIV-1 resistance'
        result = covidask.query(query)
        logger.info(f'Answers to a question: {query}')
        logger.info(f'{[r["answer"] for r in result["ret"]]}')

    elif args.run_mode == 'batch_query':
        queries = [
            'Which Lisp framework has been developed for image processing?',
            'What are the 3 main bacteria found in human milk?',
            'Where did COVID-19 happen?'
        ]
        result = covidask.batch_query(
            queries,
            max_answer_length=args.max_answer_length,
            start_top_k=args.start_top_k,
            mid_top_k=args.mid_top_k,
            top_k=args.top_k,
            doc_top_k=args.doc_top_k,
            nprobe=args.nprobe,
            sparse_weight=args.sparse_weight,
            search_strategy=args.search_strategy,
        )
        for query, result in zip(queries, result['ret']):
            logger.info(f'Answers to a question: {query}')
            logger.info(f'{[r["answer"] for r in result]}')

    elif args.run_mode == 'save_top_k':
        covidask.save_top_k(args)

    elif args.run_mode == 'eval_inmemory':
        covidask.eval_inmemory(args)

    elif args.run_mode == 'eval_request':
        covidask.eval_request(args)

    elif args.run_mode == 'eval_sent':
        covidask.eval_sent(args)

    elif args.run_mode == 'eval_sent_inm':
        covidask.eval_sent_inm(args)

    elif args.run_mode == 'eval_trec':
        covidask.eval_trec(args)

    elif args.run_mode == 'get_doc_scores':
        queries = [
            'What was the Yuan\'s paper money called?',
            'What makes a successful startup??',
            'On which date was Genghis Khan\'s palace rediscovered by archeaologists?',
            'To-y is a _ .'
        ]
        result = covidask.get_doc_scores(queries, [[36], [2], [31], [2222]])
        logger.info(result)
        result = covidask.get_top_docs(queries, 5)
        logger.info(result)
        result = covidask.get_doc_meta(['29970463']) # Only used when there's doc_meta
        logger.info(result)
        result = covidask.get_doc_meta(['COVID-ABS_418538_network-based drug repurposing for novel coronavirus 2019-ncov/sars-cov-2']) # Only used when there's doc_meta
        logger.info(result)
        result = covidask.get_q_spvecs(queries)
        logger.info(result)

    else:
        raise NotImplementedError
