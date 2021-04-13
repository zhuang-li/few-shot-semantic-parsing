import json
from nltk.corpus import stopwords
from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit, is_predicate
from grammar.action import ReduceAction, GenAction
from grammar.db import create_template_to_leaves
from grammar.rule import product_rules_to_actions
from grammar.rule import Action
from components.dataset import Example
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry
import os
import sys
from itertools import chain
import re
from preprocess_data.geo.process_geoquery import q_process
from preprocess_data.utils import generate_dir

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import operator
from grammar.rule import get_reduce_action_length_set

RATIO = 0.5
path_prefix = "../../../../datasets/geo/lambda/"
dump_path_prefix = "../../../../datasets/geo/lambda/query_split/few_shot_split_random_5_predi/"

import os
if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)

if not os.path.exists(dump_path_prefix):
    os.makedirs(dump_path_prefix)

def parse_lambda_query_helper(elem_list):
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    i = 0
    current = root
    node_pos = 1

    for elem in elem_list:
        if elem == ')':
            current = current.parent
        elif not elem in [',', '(', ';']:
            last_elem = elem_list[i - 1]
            if last_elem == '(':
                child = RuleVertex(elem)
                child.parent = current
                current.add(child)
                current = child
                child.position = node_pos
                node_pos += 1
            else:
                is_variable = is_var(elem, dataset='geo_lambda')
                is_literal = is_lit(elem, dataset='geo_lambda')
                if is_variable:
                    norm_elem = VAR_NAME
                elif is_literal:
                    norm_elem = SLOT_PREFIX
                else:
                    norm_elem = elem
                child = RuleVertex(norm_elem)
                if is_literal:
                    child.original_var = elem
                child.parent = current
                current.add(child)
                child.position = node_pos
                node_pos += 1
        i += 1

    return root


def parse_lambda_query(bracket_query):
    root = parse_lambda_query_helper(bracket_query)
    return root


def produce_data(data_filepath):
    example = []
    tgt_code_list = []
    src_list = []
    with open(data_filepath) as json_file:
        for line in json_file:
            src, tgt = line.split('\t')
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            src = src.strip()
            #src = re.sub(r"((?:c|s|m|r|co|n)\d)", VAR_NAME, src)
            src_list.append(src.split(' '))

            tgt_code_list.append(tgt.split(' '))

    tgt_asts = [parse_lambda_query(t) for t in tgt_code_list]

    temp_db = normalize_trees(tgt_asts)
    leaves_list = create_template_to_leaves(tgt_asts, temp_db)
    tid2config_seq = product_rules_to_actions(tgt_asts, leaves_list, temp_db, True, "ProductionRuleBLB",turn_v_back=False)
    reduce_action_length_set = get_reduce_action_length_set(tid2config_seq)

    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"
    assert isinstance(list(temp_db.action2id.keys())[0], Action), "action2id must contain actions"
    length_action = 0
    for i, src_sent in enumerate(src_list):
        # todo change it back
        # temp_list = [type(action).__name__ if isinstance(action, ReduceAction) else action for action in tid2config_seq[i]]

        example.append(
            Example(src_sent=src_sent, tgt_code=tgt_code_list[i], tgt_ast=tgt_asts[i], tgt_actions=tid2config_seq[i],
                    idx=i, meta=None))

        assert len(tid2config_seq[i]) == len(example[i].tgt_ast_seq), "the node head length must be equal to the action length"
    #example.sort(key=lambda x : len(x.tgt_actions))

    return example, temp_db.action2id


def split_by_new_token(query_dict, num_non_overlap = 4, random_shuffle = True):
    train_dict = {}
    test_dict = {}
    vocab_dict = {}

    for k, v in query_dict.items():
        for word in k.split(' '):
            if is_predicate(word,dataset='geo_lambda'):
                if word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1
    vocab_list = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    if random_shuffle:
        np.random.shuffle(vocab_list)
    vocab_list = vocab_list[:num_non_overlap]
    print("vocab list ", vocab_list)
    vocab_dict_freq = dict(vocab_list)
    vocab_dict_freq = {"area:i" : 1, "elevation:i": 1, "density:i" : 1 , "size:i": 1, "not" : 1}
    for k, v in query_dict.items():
        flag = True
        for word in k.split(' '):
            if word in vocab_dict_freq:
                test_dict[k] = v
                flag = False
                break
        if flag:
            train_dict[k] = v
    train_vocab_dict = {}
    for k, v in train_dict.items():
        for word in k.split(' '):
            if word in train_vocab_dict:
                train_vocab_dict[word] += 1
            else:
                train_vocab_dict[word] = 1

    return train_dict, test_dict, vocab_dict_freq


def select_support_set(query_dict, vocab_dict_freq, k_shot = 1):
    support_dict = {}
    pred_freq = {}
    query_list = list(query_dict.items())
    np.random.shuffle(query_list)
    for template, example_list in query_list:
        flag = True
        for word in template.split(' '):
            if word in vocab_dict_freq:
                if word in pred_freq:
                    if pred_freq[word] < k_shot:
                        pred_freq[word] += 1
                    else:
                        flag = False
                        break
                else:
                    pred_freq[word] = 1
                break
        if not flag:
            continue
        np.random.shuffle(example_list)
        support_dict[template] = []
        support_dict[template].append(example_list[-1])
        #example_list.pop()


    return support_dict, query_dict

def split_by_ratio(query_dict, ratio=0.5):
    return dict(list(query_dict.items())[int(len(query_dict) * ratio):]), dict(
        list(query_dict.items())[:int(len(query_dict) * ratio)])

def query_split(data_set, num_non_overlap=5, filter_one = True):
    query_dict = {}
    ori_c = 0
    for e in data_set:
        template = " ".join([str(token) for token in e.to_lambda_template.split(' ')])
        if template in query_dict:
            query_dict[template].append(e)
            ori_c += 1
        else:
            query_dict[template] = []
            query_dict[template].append(e)
            ori_c += 1
    print ("original dataset has {}".format(ori_c))
    filter_c = 0
    res_query_dict = {}
    if filter_one:
        for k, v in query_dict.items():
            if len(v) > 1:
                res_query_dict[k] = v
                filter_c += len(v)
    else:
        res_query_dict = query_dict
    print ("filtered dataset has {}".format(filter_c))

    train, test, vocab_dict_freq = split_by_new_token(res_query_dict, num_non_overlap = num_non_overlap)
    #support_set, query_set = select_support_set(test, vocab_dict_freq, k_shot)
    return train, test, vocab_dict_freq

def write_align_file(data_set, file_path):
    f = open(os.path.join(dump_path_prefix, file_path), 'w')
    for template, examples in data_set.items():
        for example in examples:
            f.write(' '.join(example.src_sent) + ' ||| ' + ' '.join([str(token) for token in example.tgt_actions]) + '\n')
    f.close()

def write_train_test_file(data_set, file_path, dump_path_prefix):
    f = open(os.path.join(dump_path_prefix, file_path), 'w')
    length = 0
    for template, examples in data_set.items():
        length += len(examples)
        for example in examples:
            f.write(' '.join(example.src_sent) + '\t' + ' '.join(example.tgt_code)+ '\n')
    f.close()
    print ("length is", length)

def prepare_geo_lambda(train_file, test_file):
    vocab_freq_cutoff = 0
    train_set, train_action2id = produce_data(train_file)
    test_set, test_action2id = produce_data(test_file)
    train_set, test, vocab_dict_freq = query_split(train_set + test_set, num_non_overlap=5)
    for i in range(5):
        for j in range(2):
            print ("=====================================================================")
            print ("Shuffle ", i)
            print ("Shot ", j+1)
            support_set, query_set = select_support_set(test, vocab_dict_freq, k_shot=j + 1)
            # train
            #write_align_file(train_set, 'align_train.txt')
            file_path = dump_path_prefix + "shuffle_{}_shot_{}".format(i, j + 1)
            generate_dir(file_path)
            print ("train set template length", len(train_set))
            write_train_test_file(train_set, 'train.txt', file_path)

            print ("support set template length", len(support_set))
            write_train_test_file(support_set, 'support.txt', file_path)
            print ("query set template length", len(query_set))
            write_train_test_file(query_set, 'query.txt', file_path)

if __name__ == '__main__':
    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    prepare_geo_lambda(train_file, test_file)
    pass
