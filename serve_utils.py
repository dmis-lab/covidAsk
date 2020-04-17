import json
import os
import pdb


def highlight_entities(splitted_content, cat, item, cName, qids=[]):
    if 'id' in item:
        if item['id'] in qids:
            cName = 'query_entities'
        if 'MESH' in item['id'] and cat == 'drug':
            _id = item['id'].split('MESH:')[1]
            item_url = "http://ctdbase.org/detail.go?type=chem?acc=" + _id
        if 'MESH' in item['id'] or 'OMIM' in item['id']:
            item_url = "http://ctdbase.org/detail.go?type=disease&acc=" + item['id']
        elif 'NCBI:txid' in item['id']:
            _id = item['id'].split('NCBI:txid')[1]
            item_url = "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=" + _id
        elif 'NCBI:gene' in item['id']:
            _id = item['id'].split('NCBI:gene')[1]
            item_url = "http://ctdbase.org/detail.go?type=gene&acc=" + _id
        else:
            return splitted_content
        
        anchor_tag = "<a href='" + item_url + "' target='_blank' class='" + cName + "'>"
        # if 'original_name' in item:
        #     splitted_content[item['start']] = anchor_tag + item['original_name']
        #     for idx in range(item['start']+1, item['end']-1):
        #         splitted_content[idx] = ''
        #     splitted_content[item['end']-1] = "</a>"
        # else:
        if item['end'] > len(splitted_content): item['end'] = len(splitted_content)
        splitted_content[item['start']] = anchor_tag + splitted_content[item['start']]
        splitted_content[item['end']-1] = splitted_content[item['end']-1] + "</a>"

    return splitted_content


def load_caches(args):
    # cache_path = {'covid': args.top100_covid_examples_path, 'google': args.top100_google_examples_path}
    cache_path = {'kcw': args.top10_examples_path}
    cached_set = {}

    for c_key, c_path in cache_path.items():
        with open(c_path, 'r') as rf:
            cache_res = json.load(rf)
            cache_res = cache_res['data']
            for k, v in cache_res.items():
                cached_set[k] = v

    return cached_set


def parse_example(args):
    index_example_set = {}
    search_examples = {}
    inverted_examples = {}
    query_entity_ids = {}

    # example_path = {'covid': args.examples_path, 'google': args.google_examples_path}
    example_path = {'kcw': args.examples_path}
    for example_key, e_path in example_path.items():

        with open(e_path, 'r') as fp:
            examples = json.load(fp)
            examples = examples['data']

            index_examples = {} # Examples rendered in index page - head question: questions list
            # search_examples = {} # Questions rendered in search page - question id: question informations
            # inverted_examples = {} # Lookup table to find question id - question lowercased text: question id
            # query_entity_ids = {} # Query ids of questions - question id: query id list

            for item in examples:
                example_id = item['id']
                example_content = item['question']

                inverted_examples[example_content.lower()] = example_id
                if example_key =='covid':
                    head_question = example_id.split('_')[1:-2]
                    head_question = ' '.join(head_question)
                    head_question = head_question[0].upper() + head_question[1:]
                elif example_key == 'kcw':
                    head_question = 'Questions'
                else:
                    head_question = example_id.split('_')[:-1]
                    head_question = ' '.join(head_question)
                    head_question = head_question[0].upper() + head_question[1:]

                parsed_example = {'id': example_id, 'content': example_content}

                highlighted_question = [char for char in example_content]
                if 'question_entities' in item.keys():
                    for q_e_cat, q_e_items in item['question_entities'].items():
                        for q_e_item in q_e_items:
                            highlighted_question = highlight_entities(
                                highlighted_question, q_e_cat, q_e_item, 'query_entities', qids = []
                            )
                            if 'id' in q_e_item:
                                if example_id in query_entity_ids.keys():
                                    query_entity_ids[example_id].append(q_e_item['id'])
                                else:
                                    query_entity_ids[example_id] = [q_e_item['id']]
                if example_id in query_entity_ids.keys():
                    query_entity_ids[example_id] = list(set(query_entity_ids[example_id]))
                highlighted_question = ''.join(highlighted_question)
                parsed_example['highlighted'] = highlighted_question

                if head_question in index_examples.keys():
                    index_examples[head_question].append(parsed_example)
                    # examples_dict[example_id].append(example_dict)
                else:
                    index_examples[head_question] = [parsed_example]
                    # examples_dict[example_id] = [example_dict]
                search_examples[example_id] = parsed_example
            index_example_set[example_key] = index_examples
    return index_example_set, search_examples, inverted_examples, query_entity_ids

def find_sublist(sl, l):
    res = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            res.append((ind, ind+sll))

    return res

def out_to_res(out, qids, cat):
    res = []
    for item in out:
        _res = {}
        _res['answer'] = item['answer']
        _res['score'] = item['score']
        _res['context'] = item['context']
        _res['title'] = item['title']
        splitted_context = [char for char in item['context']]
        if 'sent_start' in item.keys() and 'sent_end' in item.keys():
            splitted_context[item['sent_start']] = "<em>" + splitted_context[item['sent_start']]
            splitted_context[item['sent_end']-1] = splitted_context[item['sent_end']-1] + "</em>"
        if 'start_pos' in item.keys() and 'end_pos' in item.keys():
            splitted_context[item['start_pos']] = "<span class='answer_span'>" + splitted_context[item['start_pos']]
            splitted_context[item['end_pos']-1] = splitted_context[item['end_pos']-1] + "</span>"
        if 'metadata' in item.keys():
            _res['metadata'] = item['metadata']
            
            if cat == 'denspi':
                if 'paragraphs' in _res['metadata']:
                    for p in _res['metadata']['paragraphs']:
                        if 'context_entities' in p:
                            for c_e_cat, c_e_items in p['context_entities'].items():
                                for c_e_item in c_e_items:
                                    splitted_context = highlight_entities(
                                        splitted_context, c_e_cat, c_e_item, 'context_entities', qids=qids
                                    )
            elif cat == 'best':
                answer_terms = _res['answer'].split(' ')
                lowered_splitted_context = ''.join(splitted_context)
                lowered_splitted_context = lowered_splitted_context.lower()
                lowered_splitted_context = [char for char in lowered_splitted_context]
                for answer_term in answer_terms:
                    splitted_answer_term = [char for char in answer_term.lower()]
                    for sub_indices in find_sublist(splitted_answer_term, lowered_splitted_context):
                        sub_start_pos = sub_indices[0]
                        sub_end_pos = sub_indices[1]
                        splitted_context[sub_start_pos] = "<span class='best_entities'>" + splitted_context[sub_start_pos]
                        splitted_context[sub_end_pos-1] = splitted_context[sub_end_pos-1] + "</span>"
        
        if 'c_start' in item.keys():
            if item['c_start'] != 0:
                splitted_context[item['c_start']] = '... ' + splitted_context[item['c_start']]
            if item['c_end'] != len(splitted_context):
                splitted_context[item['c_end']-1] = splitted_context[item['c_end']-1] + " ..."
            splitted_context = splitted_context[item['c_start']:item['c_end']]

        _res['parsed_text'] = ''.join(splitted_context)
        res.append(_res)
    return res


def get_cached(search_examples, q_id, query_entity_ids, cached_set):
    if q_id in cached_set:
        out = cached_set[q_id]
    else:
        out = None
        return jsonify({'res': 'fail'})

    query_info = search_examples[q_id]
    query = query_info['content']
    if q_id in query_entity_ids.keys():
        qids = query_entity_ids[q_id]
    else:
        qids = []

    qids = []

    if len(out['denspi']) > 3:
        out['denspi'] = out['denspi'][:3]

    d_res = out_to_res(out['denspi'], qids, 'denspi')
    b_res = out_to_res(out['best']['ret'], qids, 'best')
    res = {'denspi': d_res, 'best': b_res}
    return res, query, query_info


def get_search(inverted_examples, search_examples, query_entity_ids, query, out, best_out):
    if query.lower() in inverted_examples.keys():
        query_id = inverted_examples[query.lower()]
        query_info = search_examples[query_id]
        if query_id in query_entity_ids.keys():
            qids = query_entity_ids[query_id]
        else:
            qids = []
    else:
        query_info = {}
        qids = []

    qids = []
    d_res = out_to_res(out['ret'], qids, 'denspi')
    b_res = out_to_res(best_out['ret'], qids, 'best')
    res = {'denspi': d_res, 'best': b_res}
    return res, query, query_info
