import collections
import json
import logging
import h5py
import os

import pandas as pd
import six
import random
import torch
import pickle as pkl
import numpy as np
import copy
import unicodedata
from tqdm import tqdm

import tokenization
from post import get_final_text_

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
NO_ANS = -1
special_stat = {}


class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id=None,
                 question_text=None,
                 paragraph_text=None,
                 doc_words=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 title="",
                 doc_idx=0,
                 par_idx=0,
                 metadata=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.doc_words = doc_words
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.title = title
        self.doc_idx = doc_idx
        self.par_idx = par_idx
        self.metadata = metadata

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(self.question_text))
        s += ", paragraph_text: %s" % (tokenization.printable_text(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class ContextFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_word_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 paragraph_index=None,
                 segment_ids=None, # Deprecated due to context/question split
                 start_position=None,
                 end_position=None,
                 switch=None,
                 answer_mask=None,
                 doc_tokens=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_word_map = token_to_word_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.switch = switch
        self.answer_mask = answer_mask
        self.paragraph_index = paragraph_index
        self.doc_tokens = doc_tokens


class QuestionFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens_,
                 input_ids,
                 input_mask,
                 segment_ids=None): # Deprecated due to context/question split
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens_ = tokens_
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def read_squad_examples(input_file, return_answers, context_only=False, question_only=False,
                        draft=False, draft_num_examples=12, append_title=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    ans_cnt = 0
    no_ans_cnt = 0

    # Only word-based tokenization is peformed (whitespace based)
    for doc_idx, entry in enumerate(input_data):
        title = entry['title'][0] if type(entry['title']) == list else entry['title']
        assert type(title) == str

        for par_idx, paragraph in enumerate(entry["paragraphs"]):
            # Do not load context for question only
            if not question_only:
                paragraph_text = paragraph["context"]
                title_offset = 0
                if append_title:
                    title_str = '[ ' + ' '.join(title.split('_')) + ' ] '
                    title_offset += len(title_str)
                    paragraph_text = title_str + paragraph_text
                # Note that we use the term 'word' for whitespace based words, and 'token' for subtokens (for BERT input)
                doc_words, char_to_word_offset = context_to_words_and_offset(paragraph_text)

            # 1) Context only ends here
            if context_only:
                metadata = {}
                if "pubmed_id" in entry:
                    entry_keys = [
                        "pubmed_id", "sha", "title_original", "title_entities",
                        "journal", "authors", "article_idx"
                    ]
                    para_keys = ["context_entities"]
                    for entry_key in entry_keys:
                        if entry_key in entry:
                            metadata[entry_key] = entry[entry_key]
                    for para_key in para_keys:
                        if para_key in paragraph:
                            metadata[para_key] = paragraph[para_key]
                    # metadata["pubmed_id"] = (metadata["pubmed_id"] if not pd.isnull(metadata["pubmed_id"])
                    #     else 'NaN')
                example = SquadExample(
                    doc_words=doc_words,
                    title=title,
                    doc_idx=doc_idx,
                    par_idx=par_idx,
                    metadata=metadata)
                examples.append(example)

                if draft and len(examples) == draft_num_examples:
                    return examples
                continue

            # 2) Question only or 3) context/question pair
            else:
                for qa in paragraph["qas"]:
                    qas_id = str(qa["id"])
                    question_text = qa["question"]

                    # Noisy question skipping
                    if len(question_text.split(' ')) == 1:
                        logger.info('Skipping a single word question: {}'.format(question_text))
                        continue
                    if "I couldn't could up with another question." in question_text:
                        logger.info('Skipping a strange question: {}'.format(question_text))
                        continue

                    start_position = None
                    end_position = None
                    orig_answer_text = None

                    # For pre-processing that should return answers together
                    if return_answers:
                        assert type(qa["answers"]) == dict or type(qa["answers"]) == list, type(qa["answers"])
                        if type(qa["answers"]) == dict:
                            qa["answers"] = [qa["answers"]]

                        # No answers
                        if len(qa["answers"]) == 0:
                            orig_answer_text = ""
                            start_position = -1 # Word-level no-answer => -1
                            end_position = -1
                            no_ans_cnt += 1
                        # Answer exists
                        else:
                            answer = qa["answers"][0]
                            ans_cnt += 1

                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"] + title_offset
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]

                            # Only add answers where the text can be exactly recovered from the context
                            actual_text = " ".join(doc_words[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                tokenization.whitespace_tokenize(orig_answer_text)) # word based tokenization
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue

                    # Question only ends here
                    if question_only:
                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text)

                    # Context/question pair ends here
                    else:
                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            paragraph_text=paragraph_text,
                            doc_words=doc_words,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            title=title,
                            doc_idx=doc_idx,
                            par_idx=par_idx)
                    examples.append(example)

                    if draft and len(examples) == draft_num_examples:
                        return examples

    # Testing for shuffled draft (should comment out above 'draft' if-else statements)
    if draft:
        random.shuffle(examples)
        logger.info(str(len(examples)) + ' were collected before draft for shuffling')
        return examples[:draft_num_examples]

    logger.info('Answer/no-answer stat: %d vs %d'%(ans_cnt, no_ans_cnt))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, return_answers, skip_no_answer,
                                 verbose=False, save_with_prob=False, msg="Converting examples"):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []
    question_features = []

    for (example_index, example) in enumerate(tqdm(examples, desc=msg)):

        # Tokenize query into (sub)tokens
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # Creating a map between word <=> (sub)token
        tok_to_word_index = []
        word_to_tok_index = [] # word to (start of) subtokens
        all_doc_tokens = []
        for (i, word) in enumerate(example.doc_words):
            word_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                tok_to_word_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -2 accounts for [CLS], [SEP]
        max_tokens_for_doc = max_seq_length - 2

        # Split sequence by max_seq_len with doc_stride, _DocSpan is based on tokens without [CLS], [SEP]
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_tok_offset = 0 # From all_doc_tokens

        # Get doc_spans with stride and offset
        while start_tok_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_tok_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_tok_offset, length=length))
            if start_tok_offset + length == len(all_doc_tokens):
                break
            start_tok_offset += min(length, doc_stride) # seems to prefer doc_stride always
            assert doc_stride < length, "length is no larger than doc_stride for {}".format(doc_spans)

        # Iterate each doc_span and make out_tokens
        for (doc_span_index, doc_span) in enumerate(doc_spans):

            # Find answer position based on new out_tokens
            start_position = None
            end_position = None

            # For no_answer, same (-1, -1) applies
            if example.start_position is not None and example.start_position < 0:
                assert example.start_position == -1 and example.end_position == -1
                start_position, end_position = NO_ANS, NO_ANS

            # For existing answers, find answers if exist
            elif return_answers:

                # Get token-level start/end position
                tok_start_position = word_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_words) - 1:
                    tok_end_position = word_to_tok_index[example.end_position + 1] - 1 # By backwarding from next word
                else:
                    assert example.end_position == len(example.doc_words) - 1
                    tok_end_position = len(all_doc_tokens) - 1

                # Improve answer span by subword-level
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

                # Throw away training samples without answers (due to doc_span split)
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (tok_start_position < doc_start or tok_end_position < doc_start or
                    tok_start_position > doc_end or tok_end_position > doc_end):
                    if skip_no_answer:
                        continue
                    else:
                        # For NQ, only add this in 2% (50 times downsample)
                        if save_with_prob:
                            if np.random.randint(100) < 2:
                                start_position, end_position = NO_ANS, NO_ANS
                            else:
                                continue
                        else:
                            start_position, end_position = NO_ANS, NO_ANS

                # Training samples with answers
                else:
                    doc_offset = 1 # For [CLS]
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    assert start_position >= 0 and end_position >= 0, (start_position, end_position)

            out_tokens = []  # doc
            out_tokens_ = [] # quesry
            out_tokens.append("[CLS]")
            out_tokens_.append("[CLS]")
            token_to_word_map = {} # The difference with tok_to_word_index is it includes special tokens
            token_is_max_context = {}

            # For query tokens, just copy and add [SEP]
            for token in query_tokens:
                out_tokens_.append(token)
            out_tokens_.append("[SEP]")

            # For each doc token, create token_to_word_map and is_max_context, and add to out_tokens
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_word_map[len(out_tokens)] = tok_to_word_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(out_tokens)] = is_max_context
                out_tokens.append(all_doc_tokens[split_token_index])
            out_tokens.append("[SEP]")

            # Convert to ids and masks
            input_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            input_ids_ = tokenizer.convert_tokens_to_ids(out_tokens_)
            input_mask = [1] * len(input_ids)
            input_mask_ = [1] * len(input_ids_)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            while len(input_ids_) < max_query_length + 2: # +2 for [CLS], [SEP]
                input_ids_.append(0)
                input_mask_.append(0)
            assert len(input_ids_) == max_query_length + 2
            assert len(input_mask_) == max_query_length + 2

            # Printing for debug
            if example_index < 1 and verbose:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in out_tokens]))
                logger.info("q tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in out_tokens_]))
                logger.info("token_to_word_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_word_map)]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                if return_answers:
                    answer_text = " ".join(out_tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            # Append feature
            features.append(
                ContextFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=out_tokens,
                    token_to_word_map=token_to_word_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    start_position=start_position,
                    end_position=end_position))
            question_features.append(
                QuestionFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens_=out_tokens_,
                    input_ids=input_ids_,
                    input_mask=input_mask_))

            # Check validity of answer
            if return_answers:
                if start_position <= NO_ANS:
                    assert start_position == NO_ANS and end_position == NO_ANS, (start_position, end_position)
                else:
                    assert out_tokens[start_position:end_position+1] == \
                            all_doc_tokens[tok_start_position:tok_end_position+1]
                    orig_text, start_pos, end_pos = get_final_text_(
                        example, features[-1], start_position, end_position, True, False)
                    phrase = orig_text[start_pos:end_pos]
                    try:
                        assert phrase == example.orig_answer_text
                    except Exception as e:
                        # print('diff ans [%s]/[%s]'%(phrase, example.orig_answer_text))
                        pass
            unique_id += 1

    return features, question_features


def convert_questions_to_features(examples, tokenizer, max_query_length=None):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    question_features = []

    for (example_index, example) in enumerate(tqdm(examples, desc='Converting questions')):

        query_tokens = tokenizer.tokenize(example.question_text)
        if max_query_length is None:
            max_query_length = len(query_tokens)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        for _ in enumerate(range(1)):
            tokens_ = []
            tokens_.append("[CLS]")
            for token in query_tokens:
                tokens_.append(token)
            tokens_.append("[SEP]")

            input_ids_ = tokenizer.convert_tokens_to_ids(tokens_)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask_ = [1] * len(input_ids_)

            # Zero-pad up to the sequence length.
            while len(input_ids_) < max_query_length + 2:
                input_ids_.append(0)
                input_mask_.append(0)

            assert len(input_ids_) == max_query_length + 2
            assert len(input_mask_) == max_query_length + 2

            if example_index < 1:
                # logger.info("*** Example ***")
                # logger.info("unique_id: %s" % (unique_id))
                # logger.info("example_index: %s" % (example_index))
                logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in query_tokens]))
                # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_]))
                # logger.info(
                #     "input_mask: %s" % " ".join([str(x) for x in input_mask_]))

            question_features.append(
                QuestionFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens_=tokens_,
                    input_ids=input_ids_,
                    input_mask=input_mask_))
            unique_id += 1

    return question_features


def convert_documents_to_features(examples, tokenizer, max_seq_length, doc_stride):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(tqdm(examples, desc='Converting documents')):

        # Creating a map between word <=> (sub)token
        tok_to_word_index = []
        word_to_tok_index = [] # word to (start of) subtokens
        all_doc_tokens = []
        for (i, word) in enumerate(example.doc_words):
            word_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                tok_to_word_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -2 accounts for [CLS], [SEP]
        max_tokens_for_doc = max_seq_length - 2

        # Split sequence by max_seq_len with doc_stride, _DocSpan is based on tokens without [CLS], [SEP]
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_tok_offset = 0 # From all_doc_tokens

        # Get doc_spans with stride and offset
        while start_tok_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_tok_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_tok_offset, length=length))
            if start_tok_offset + length == len(all_doc_tokens):
                break
            start_tok_offset += min(length, doc_stride) # seems to prefer doc_stride always
            assert doc_stride < length, "length is no larger than doc_stride for {}".format(doc_spans)

        # Iterate each doc_span and make out_tokens
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            out_tokens = []  # doc
            out_tokens.append("[CLS]")
            token_to_word_map = {} # The difference with tok_to_word_index is it includes special tokens
            token_is_max_context = {}

            # For each doc token, create token_to_word_map and is_max_context, and add to out_tokens
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_word_map[len(out_tokens)] = tok_to_word_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(out_tokens)] = is_max_context
                out_tokens.append(all_doc_tokens[split_token_index])
            out_tokens.append("[SEP]")

            # Convert to ids and masks
            input_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            # Printing for debug
            if example_index < 1 and doc_span_index < 1:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in out_tokens]))
                logger.info("token_to_word_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_word_map)]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))

            # Append feature
            features.append(
                ContextFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=out_tokens,
                    token_to_word_map=token_to_word_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask))
            unique_id += 1

    return features


def context_to_words_and_offset(context):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_words = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_words.append(c)
            else:
                doc_words[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_words) - 1)

    return doc_words, char_to_word_offset


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
