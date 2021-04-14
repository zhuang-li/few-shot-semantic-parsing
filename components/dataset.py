# coding=utf-8
from collections import namedtuple

import numpy as np
import pickle

import torch

from grammar.vertex import RuleVertex, CompositeTreeVertex
from model import nn_utils
from grammar.rule import extract_action_lit
from common.utils import cached_property
from grammar.action import *
import random
from math import gcd
import operator
from collections import Counter
from grammar.utils import is_var
from grammar.consts import VAR_NAME, IMPLICIT_HEAD, ROOT
import re

class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @property
    def all_actions(self):
        return [e.tgt_actions for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    @staticmethod
    def from_bin_file_list(file_path_list):
        examples = []
        for file_path in file_path_list:
            examples.extend(pickle.load(open(file_path, 'rb')))
        return Dataset(examples)

    def add(self, new_dataset):
        self.examples.extend(new_dataset.examples)

    def add_examples(self, new_examples):
        self.examples.extend(new_examples)

    def batch_iter(self, batch_size, shuffle=False, sort=True):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            if sort:
                batch_examples.sort(key=lambda e: -len(e.src_sent))

            yield batch_examples

    def random_sample_batch_iter(self, sample_size):
        index_arr = np.arange(len(self.examples))
        np.random.shuffle(index_arr)
        batch_ids = index_arr[:sample_size]
        batch_examples = [self.examples[i] for i in batch_ids]

        return batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class EposideDataset(object):
    def __init__(self, examples):
        self.examples = examples
        self.template_instances = self._read_by_templates(examples)

    def _read_by_templates(self, examples):
        template_instances = dict()
        for example in examples:
            template = " ".join([str(action) for action in example.tgt_actions])
            if template in template_instances:
                template_instances[template].append(example)
            else:
                template_instances[template] = []
                template_instances[template].append(example)
        return template_instances

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @property
    def all_actions(self):
        return [e.tgt_actions for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return EposideDataset(examples)

    def batch_iter(self, n_way, k_shot, query_num, shuffle=False):
        index_arr = np.arange(len(self.template_instances.keys()))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.template_instances.keys()) / float(n_way)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[n_way * batch_id: n_way * (batch_id + 1)]
            template_examples = [list(self.template_instances.keys())[i] for i in batch_ids]
            template_examples.sort(key=lambda template: -len(template.split(' ')))
            support_batch = []
            query_batch = []
            for template in template_examples:
                type_examples = self.template_instances[template]
                rand_inds = torch.randperm(len(type_examples))
                support_inds = rand_inds[:k_shot]
                query_inds = rand_inds[k_shot:k_shot+query_num]
                xs = [type_examples[s_index] for s_index in support_inds.tolist()]
                xq = [type_examples[q_index] for q_index in query_inds.tolist()]
                support_batch.extend(xs)
                query_batch.extend(xq)

            support_batch.sort(key=lambda e: -len(e.src_sent))
            query_batch.sort(key=lambda e: -len(e.src_sent))
            yield support_batch, query_batch

    def task_batch_iter(self, n_way, k_shot, query_num, task_num = 4, shuffle=False):
        for i in range(task_num):
            index_arr = np.arange(len(self.template_instances.keys()))
            if shuffle:
                np.random.shuffle(index_arr)

            batch_ids = index_arr[:n_way]
            template_examples = [list(self.template_instances.keys())[i] for i in batch_ids]
            template_examples.sort(key=lambda template: -len(template.split(' ')))
            support_batch = []
            query_batch = []
            for template in template_examples:
                type_examples = self.template_instances[template]
                rand_inds = torch.randperm(len(type_examples))

                support_inds = rand_inds[:k_shot]
                query_inds = rand_inds[k_shot:k_shot+query_num]
                xs = [type_examples[s_index] for s_index in support_inds.tolist()]
                xq = [type_examples[q_index] for q_index in query_inds.tolist()]

                support_batch.extend(xs)
                query_batch.extend(xq)
            support_batch.sort(key=lambda e: -len(e.src_sent))
            query_batch.sort(key=lambda e: -len(e.src_sent))
            yield support_batch, query_batch

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        # str list
        self.src_sent = src_sent
        # str list
        self.tgt_code = tgt_code
        # vertext root
        self.tgt_ast = tgt_ast
        # action sequence
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return " ".join(self.src_sent)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def copy(self):
        new_src_sent = [token for token in self.src_sent]
        new_tgt_code = [token for token in self.tgt_code]
        new_tgt_ast = self.tgt_ast.copy()
        new_tgt_actions = [action.copy() for action in self.tgt_actions]
        new_idx = self.idx
        new_meta = self.meta

        return Example(new_src_sent, new_tgt_actions ,new_tgt_code,  new_tgt_ast, idx=new_idx, meta=new_meta)


    @property
    def to_prolog_template(self):
        return self.tgt_ast.to_prolog_expr

    @property
    def to_lambda_template(self):
        return self.tgt_ast.to_lambda_expr


    @property
    def to_logic_form(self):
        tgt_code = list(self.tgt_code)
        if not (self.tgt_code[0] == '(' and self.tgt_code[-1] == ')'):
            tgt_code = ['('] + tgt_code + [')']
        return " ".join([str(code) for code in tgt_code])

    @property
    def tgt_ast_seq(self):
        tgt_ast_seq = []
        for action in self.tgt_actions:
            if isinstance(action, GenAction):
                tgt_ast_seq.append(action.vertex)
            elif isinstance(action, ReduceAction):
                tgt_ast_seq.append(action.rule.head)
            else:
                raise ValueError

        return tgt_ast_seq

class Batch(object):
    def __init__(self, examples, vocab, training=True, append_boundary_sym=True, use_cuda=False):
        self.examples = examples

        # source token seq
        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) + 2 if append_boundary_sym else len(e.src_sent) for e in self.examples]

        # target token seq
        self.tgt_code = [e.tgt_code for e in self.examples]
        self.tgt_code_len = [len(e.tgt_code) + 2 if append_boundary_sym else len(e.tgt_code) for e in self.examples]

        # action seq

        self.action_seq = [e.tgt_actions for e in self.examples]
        self.action_seq_len = [len(e.tgt_actions) + 1 if append_boundary_sym else len(e.tgt_actions) for e in
                               self.examples]
        self.max_action_num = max(self.action_seq_len)

        self.max_ast_seq_num = max(len(e.tgt_ast_seq) for e in self.examples)
        self.ast_seq = [e.tgt_ast_seq for e in self.examples]
        self.ast_seq_len = [len(e.tgt_ast_seq) + 1 if append_boundary_sym else len(e.tgt_ast_seq) for e in
                               self.examples]

        self.entity_seq = []
        for seq in self.action_seq:
            self.entity_seq.append([])
            for action in seq:
                self.entity_seq[-1].extend(action.entities)

        self.variable_seq = []
        for seq in self.action_seq:
            self.variable_seq.append([])
            for action in seq:
                self.variable_seq[-1].extend(action.variables)

        self.vocab = vocab
        self.use_cuda = use_cuda
        self.training = training
        self.append_boundary_sym = append_boundary_sym


    def __len__(self):
        return len(self.examples)



    # source sentence
    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')



    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    use_cuda=self.use_cuda)

    # action sequence
    @cached_property
    def action_seq_var(self):
        #print (self.action_seq)
        return nn_utils.to_input_variable(self.action_seq, self.vocab.action,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='action')

    # action sequence
    @cached_property
    def action_seq_pad(self):
        return nn_utils.to_seq_pad(self.action_seq, append_boundary_sym=True, mode ='action')

    @cached_property
    def action_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.action_seq_len,
                                                    use_cuda=self.use_cuda)

    # general action sequence
    @cached_property
    def general_action_seq_var(self):

        return nn_utils.to_input_variable(self.action_seq, self.vocab.general_action,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='action')

    @cached_property
    def general_action_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.action_seq_len,
                                                    use_cuda=self.use_cuda)

    # target sequence
    @cached_property
    def tgt_seq_var(self):
        return nn_utils.to_input_variable(self.tgt_code, self.vocab.code,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')

    @cached_property
    def tgt_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.tgt_code_len,
                                                    use_cuda=self.use_cuda)

    # ast sequence
    @cached_property
    def ast_seq_var(self):
        return nn_utils.to_input_variable(self.ast_seq, self.vocab.vertex,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='ast')

    @cached_property
    def ast_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.ast_seq_len, use_cuda=self.use_cuda)

    @cached_property
    def entity_seq_var(self):
        return nn_utils.to_input_variable(self.entity_seq, self.vocab.entity,
                                          use_cuda=self.use_cuda, append_boundary_sym=True,
                                          mode='lit')

    @cached_property
    def variable_seq_var(self):
        return nn_utils.to_input_variable(self.variable_seq, self.vocab.variable,
                                          use_cuda=self.use_cuda, append_boundary_sym=True,
                                          mode='lit')