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
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits, embedding, softmax

from transformers import BertPreTrainedModel, BertModel

NO_ANS = -1


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SparseAttention(nn.Module):
    def __init__(self, config, num_sparse_heads):
        super(SparseAttention, self).__init__()
        if config.hidden_size % num_sparse_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, num_sparse_heads))
        self.num_attention_heads = num_sparse_heads
        self.attention_head_size = int(config.hidden_size / num_sparse_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.shifted_gelu = lambda x: gelu(x) + 0.2 # makes all positive

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, input_ids, ngram=1):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [B, h, T, d/h]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [B, h, T, d/h]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, h, T, T]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(-1)

        # Normalize the attention scores to probabilities.
        # attention_probs = gelu(attention_scores)
        attention_probs = self.relu(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs.transpose(1, 2)  # [B, T, h, T]
        return context_layer


class DenSPI(BertPreTrainedModel):
    def __init__(self,
                 config,
                 span_vec_size=64,
                 context_layer_idx=-1,
                 question_layer_idx=-1,
                 sparse_ngrams=['1', '2'],
                 use_sparse=True):
        super().__init__(config)

        # Dense modules
        self.bert = BertModel(config)
        self.bert_q = self.bert

        # Sparse modules
        self.sparse_start = nn.ModuleDict({
            key: SparseAttention(config, num_sparse_heads=1)
            for key in sparse_ngrams
        })
        self.sparse_end = nn.ModuleDict({
            key: SparseAttention(config, num_sparse_heads=1)
            for key in sparse_ngrams
        })
        self.sparse_start_q = self.sparse_start
        self.sparse_end_q =  self.sparse_end

        # Other parameters
        # self.linear = nn.Linear(config.hidden_size, 2) # For filter
        self.linear = nn.Linear(config.hidden_size, config.hidden_size) # For DenSPI-Sparc
        self.default_value = nn.Parameter(torch.randn(1))

        # Arguments
        self.span_vec_size = span_vec_size
        self.context_layer_idx = context_layer_idx
        self.question_layer_idx = question_layer_idx
        self.use_sparse = use_sparse
        self.sparse_ngrams = sparse_ngrams
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids=None, input_mask=None, query_ids=None, query_mask=None):

        if input_ids is not None:
            bs, seq_len = input_ids.size()

            # Get dense reps
            context_layer_last, _ = self.bert(input_ids, attention_mask=input_mask, token_type_ids=None)
            # context_layer_all = context_layers[self.context_layer_idx]

            # Calculate dense logits
            context_layer = context_layer_last[:, :, :-self.span_vec_size]
            span_layer = context_layer_last[:, :, -self.span_vec_size:]
            start, end, = context_layer.chunk(2, dim=2)
            span_start, span_end = span_layer.chunk(2, dim=2)
            span_logits = span_start.matmul(span_end.transpose(1, 2))

            # Get sparse reps
            start_sps = {}
            end_sps = {}
            sparse_mask = (input_ids >= 999).float()
            input_diag = (1 - torch.diag(torch.ones(input_ids.shape[1]))).to(sparse_mask.get_device())
            for ngram in self.sparse_ngrams:
                start_sps[ngram] = self.sparse_start[ngram](
                    context_layer_last,
                    (1 - input_mask).float() * -1e9,
                    input_ids, ngram=self.sparse_ngrams,
                )
                end_sps[ngram] = self.sparse_end[ngram](
                    context_layer_last,
                    (1 - input_mask).float() * -1e9,
                    input_ids, ngram=self.sparse_ngrams,
                )
                start_sps[ngram] = start_sps[ngram][:,:,0,:] * sparse_mask.unsqueeze(2) * input_diag.unsqueeze(0)
                end_sps[ngram] = end_sps[ngram][:,:,0,:] * sparse_mask.unsqueeze(2) * input_diag.unsqueeze(0)

            # Filter logits
            filter_start_logits, filter_end_logits = self.linear(context_layer_last).chunk(2, dim=2)
            # filter_start_logits = filter_start_logits.squeeze(2)
            # filter_end_logits = filter_end_logits.squeeze(2)
            filter_start_logits = filter_start_logits[:,:,0]
            filter_end_logits = filter_end_logits[:,:,0]

            # Embed context
            if query_ids is None:
                return start, end, span_logits, filter_start_logits, filter_end_logits, start_sps, end_sps

        if query_ids is not None:
            # Get dense reps
            question_layer_last, _ = self.bert_q(query_ids, attention_mask=query_mask, token_type_ids=None)
            question_layer = question_layer_last[:, :, :-self.span_vec_size]
            query_start, query_end = question_layer[:, :1, :].chunk(2, dim=2)  # Just [CLS]

            # Get sparse reps
            q_start_sps = {}
            q_end_sps = {}
            query_sparse_mask = ((query_ids >= 999) & (query_ids != 1029)).float()
            for ngram in self.sparse_ngrams:
                q_start_sps[ngram] = self.sparse_start_q[ngram](
                    question_layer_last,
                    (1 - query_mask).float() * -1e9,
                    query_ids, ngram=self.sparse_ngrams
                )
                q_end_sps[ngram] = self.sparse_end_q[ngram](
                    question_layer_last,
                    (1 - query_mask).float() * -1e9,
                    query_ids, ngram=self.sparse_ngrams
                )
                q_start_sps[ngram] = q_start_sps[ngram][:,0,0,:] * query_sparse_mask
                q_end_sps[ngram] = q_end_sps[ngram][:,0,0,:] * query_sparse_mask

            # Embed question
            if input_ids is None:
                return query_start, query_end, q_start_sps, q_end_sps
