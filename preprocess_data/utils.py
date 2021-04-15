import json
from nltk.corpus import stopwords
from torch import nn

from grammar.vertex import RuleVertex, CompositeTreeVertex
from grammar.consts import *
from grammar.db import normalize_trees, TemplateDB
from grammar.utils import is_var, is_lit, is_predicate
from grammar.action import ReduceAction, GenAction
from grammar.db import create_template_to_leaves
from grammar.rule import product_rules_to_actions_bottomup
from grammar.rule import Action
from components.dataset import Example, Batch
from components.vocab import Vocab, LitEntry
from model.utils import GloveHelper
from model.nn_utils import to_input_variable
import torch
from similarity.normalized_levenshtein import NormalizedLevenshtein

from preprocess_data.utils import *
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry
import os
import sys
from itertools import chain
import re
import nltk

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

def prepare_data(train_set, train_db, support_file, query_file, dump_path_prefix, data_type='atis_lambda', lang='lambda', place_holder = 6, rule_type='ProductionRuleBL'):

    train_src_list = [e.src_sent for e in train_set]
    train_action_list = [e.tgt_actions for e in train_set]

    support_set, support_db = produce_data(support_file, data_type, lang, rule_type=rule_type, normlize_tree=True, train_src_list=train_src_list, train_action_seq=train_action_list)
    query_set, query_db = produce_data(query_file, data_type, lang,rule_type=rule_type,normlize_tree=True)


    train_vocab = pre_few_shot_vocab(data_set=train_set,
                                     entity_list=train_db.entity_list + support_db.entity_list,
                                     variable_list=train_db.variable_list + support_db.variable_list, place_holder=place_holder)
    support_vocab = pre_few_shot_vocab(data_set=train_set + support_set,
                                       entity_list=train_db.entity_list + support_db.entity_list,
                                       variable_list=train_db.variable_list + support_db.variable_list)
    print('generated train vocabulary %s' % repr(train_vocab), file=sys.stderr)
    print('generated support vocabulary %s' % repr(support_vocab), file=sys.stderr)
    action_len = [len(e.tgt_actions) for e in chain(train_set, support_set, query_set)]
    print('Train set len: %d' % len(train_set))
    print('Support set len: %d' % len(support_set))
    print('Query set len: %d' % len(query_set))

    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    pickle.dump(support_set, open(os.path.join(dump_path_prefix, 'support.bin'), 'wb'))
    pickle.dump(query_set, open(os.path.join(dump_path_prefix, 'query.bin'), 'wb'))
    pickle.dump(train_vocab, open(os.path.join(dump_path_prefix, 'train_vocab.bin'), 'wb'))
    pickle.dump(support_vocab, open(os.path.join(dump_path_prefix, 'support_vocab.bin'), 'wb'))
    return support_set, query_set


def generate_examples(dataset = 'jobs', data_type = 'job_prolog', lang = 'prolog', place_holder = 3, frequency_list=[]):
    for frequency in frequency_list:
        if frequency > 0:
            rule_type = "ProductionRuleBL"
        else:
            rule_type = "ProductionRuleBLB"
        train_file = "datasets/" + dataset + "/lemma_data/shuffle_0_shot_1/train.txt"
        train_set, train_db = produce_data(train_file,  data_type=data_type, lang=lang, rule_type=rule_type, normlize_tree=True,
                                           frequent=frequency)

        for s in range(6):
            for k in range(2):
                shuffle = s
                shot = k + 1
                path_prefix = "datasets/" + dataset + "/lemma_data/shuffle_{}_shot_{}".format(shuffle, shot)
                dump_path_prefix = "datasets/" + dataset + "/freq_{}/shuffle_{}_shot_{}".format(frequency, shuffle, shot)
                generate_dir(dump_path_prefix)

                support_file = os.path.join(path_prefix, 'support.txt')
                print (support_file)
                query_file = os.path.join(path_prefix, 'query.txt')
                print (query_file)
                support_set, query_set = prepare_data(train_set, train_db, support_file, query_file, dump_path_prefix, data_type=data_type, lang=lang, place_holder = place_holder, rule_type=rule_type)



normalized_levenshtein = NormalizedLevenshtein()


def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pre_few_shot_vocab(data_set, entity_list, variable_list, place_holder=None):

    vocab_freq_cutoff = 0
    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in data_set], size=5000, freq_cutoff=vocab_freq_cutoff)

    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in data_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    entity_vocab = LitEntry.from_corpus(entity_list, size=5000, freq_cutoff=0)

    variable_vocab = LitEntry.from_corpus(variable_list, size=5000, freq_cutoff=0)

    action_vocab = ActionVocabEntry.from_corpus([e.tgt_actions for e in data_set], size=5000, freq_cutoff=0)

    general_action_vocab = GeneralActionVocabEntry.from_action_vocab(action_vocab.token2id)
    gen_vocab = GenVocabEntry.from_corpus([e.tgt_actions for e in data_set], size=5000, freq_cutoff=0)
    reduce_vocab = ReVocabEntry.from_corpus([e.tgt_actions for e in data_set], size=5000, freq_cutoff=0)
    vertex_vocab = VertexVocabEntry.from_example_list(data_set)
    if place_holder:
        for i in range(place_holder):
            action_vocab.add(GenAction(RuleVertex('<place_holder_{}>'.format(i))))
            vertex_vocab.add(RuleVertex('<place_holder_{}>'.format(i)))

    vocab = Vocab(source=src_vocab,
                  code=code_vocab,
                  action=action_vocab,
                  general_action=general_action_vocab,
                  gen_action=gen_vocab,
                  re_action=reduce_vocab,
                  vertex=vertex_vocab,
                  entity=entity_vocab,
                  variable=variable_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)
    return vocab


def recursive_get_predicate(vertex, predicate_list):
    if vertex.has_children():
        predicate_list.append(vertex.head)
        for child in vertex.children:
            if not (child.head == SLOT_PREFIX or child.head == VAR_NAME):
                recursive_get_predicate(child, predicate_list)
    else:
        predicate_list.append(vertex.head)

    return predicate_list


def get_action_predicate(action):
    predicate = []
    if isinstance(action, ReduceAction):
        vertex = action.rule.head
        if isinstance(vertex, CompositeTreeVertex):
            vertex = vertex.vertex
        predicate.extend([vertex.head])
    elif isinstance(action, GenAction):
        vertex = action.vertex
        if isinstance(vertex, CompositeTreeVertex):
            vertex = vertex.vertex
        predicate.extend(recursive_get_predicate(vertex, []))
    else:
        raise ValueError
    return predicate


def get_predicate_tokens(predicate_list):
    turned_predicate_tokens = []
    for predicate in predicate_list:
        predicate_tokens = predicate.strip().split(':')[0].split('_')
        for token in predicate_tokens:
            if token in PREDICATE_LEXICON:
                turned_predicate_tokens.extend(PREDICATE_LEXICON[token])
            else:
                turned_predicate_tokens.append(token)
    return turned_predicate_tokens


def get_entity_variable_list(tgt_code_list, data_type, temp_db):
    for tgt_list in tgt_code_list:
        temp_entity_list = []
        temp_variable_list = []
        for token in tgt_list:
            if is_var(token, dataset=data_type):
                temp_variable_list.append(token)
            elif is_lit(token, dataset=data_type):
                temp_entity_list.append(token)
        temp_db.entity_list.append(temp_entity_list)
        temp_db.variable_list.append(temp_variable_list)


def parse_lambda_query_helper(elem_list, data_type):
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    i = 0
    current = root
    node_pos = 1
    var_id = 0
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
                is_variable = is_var(elem, dataset=data_type)
                is_literal = is_lit(elem, dataset=data_type)
                if is_variable:
                    norm_elem = VAR_NAME
                elif is_literal:
                    norm_elem = SLOT_PREFIX
                    var_id += 1
                else:
                    norm_elem = elem
                child = RuleVertex(norm_elem)
                if is_variable:
                    child.original_var = elem
                elif is_literal:
                    child.original_entity = elem

                child.parent = current
                current.add(child)
                child.position = node_pos
                node_pos += 1
        i += 1

    return root


def parse_lambda_query(elem_list, data_type):
    root = parse_lambda_query_helper(elem_list, data_type)
    return root


def parse_prolog_query_helper(elem_list, data_type):
    separator_set = set([',', '(', ')', ';', '\+'])
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    depth = 0
    i = 0
    current = root
    node_pos = 1
    var_id = 0
    for elem in elem_list:
        if elem == '(':
            depth += 1
            if i > 0:
                last_elem = elem_list[i - 1]
                if last_elem in separator_set:
                    if last_elem == '\+':
                        depth += 1
                        current = current.children[-1]
                    child = RuleVertex(IMPLICIT_HEAD)
                    child.parent = current
                    current.add(child)
                    child.is_auto_nt = True

                    current = child
                    child.position = node_pos
                    node_pos += 1

                else:
                    current = current.children[-1]

        elif elem == ')':
            current = current.parent
            depth -= 1
            if current and current.head == '\+':
                current = current.parent
                depth -= 1
        elif not elem == ',':
            if i > 0:
                last_elem = elem_list[i - 1]
                if last_elem == '\+':
                    depth += 1
                    current = current.children[-1]
            is_literal = is_lit(elem, dataset=data_type)
            is_variable = is_var(elem, dataset=data_type)
            if is_literal:
                norm_elem = SLOT_PREFIX
                var_id += 1
            elif is_variable:
                norm_elem = VAR_NAME
            else:
                norm_elem = elem

            child = RuleVertex(norm_elem)
            if is_variable:
                child.original_var = elem
            elif is_literal:
                child.original_entity = elem

            child.parent = current
            current.add(child)
            child.position = node_pos
            node_pos += 1
        i += 1

    return root

def parse_prolog_query(elem_list, data_type):
    root = parse_prolog_query_helper(elem_list, data_type)
    return root

def update_cooccur_table(cooc_table, src_list, predi_freq, token):
    for src_token in src_list:
        if src_token in predi_freq:
            predi_freq[src_token] = predi_freq[src_token] + 1
        else:
            predi_freq[src_token] = 1

        if src_token in cooc_table[token]:
            cooc_table[token][src_token] = cooc_table[token][src_token] + 1
        else:
            cooc_table[token][src_token] = 1


def update_predicate_condition_table(src_list, tgt_list, predi_freq, cooc_table):
    for token in tgt_list:
        if token in cooc_table:
            update_cooccur_table(cooc_table, src_list, predi_freq, token)
        else:
            cooc_table[token] = {}
            update_cooccur_table(cooc_table, src_list, predi_freq, token)


def get_cond_score(src_sent, predicate, predicate_freq, coor_table):
    cond_score = []
    for token in src_sent:
        cond_score.append(coor_table[predicate][token] / predicate_freq[token])
    cond_score = [0] + cond_score + [0]
    return cond_score


def get_string_sim(src_sent, prototype_tokens):
    sim_score = []
    ignore_tokens = set(['A','e','var'])
    for src_token in src_sent:
        score_list = [0]
        for tgt_token in prototype_tokens:
            if tgt_token in ignore_tokens:
                score_list.append(0)
            else:
                score_list.append(1 - normalized_levenshtein.distance(src_token, tgt_token))
        sim_score.append(max(score_list))
    sim_score = [0] + sim_score + [0]

    return sim_score



def produce_data(data_filepath, data_type, lang, turn_v_back=False, normlize_tree=True, rule_type="ProductionRuleBLB", train_src_list = [], train_action_seq = [], frequent = 0):
    example = []
    tgt_code_list = []
    src_list = []
    ori_tgt_code_list = []
    with open(data_filepath) as json_file:
        for line in json_file:
            line_list = line.split('\t')
            src = line_list[0]
            tgt = line_list[1]
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            src = src.strip()
            src_split = src.split(' ')
            src_list.append(src_split)
            tgt_split = tgt.split(' ')
            ori_tgt_code_list.append(tgt_split)
            if len(tgt_split) == 1:
                tgt_split = ["(", PAD_PREDICATE] + tgt_split + [")"]
            tgt_code_list.append(tgt_split)

    json_file.close()
    if lang == 'lambda':
        tgt_asts = [parse_lambda_query(t, data_type) for t in tgt_code_list]
    elif lang == 'prolog':
        tgt_asts = [parse_prolog_query(t, data_type) for t in tgt_code_list]

    temp_db = normalize_trees(tgt_asts)
    get_entity_variable_list(tgt_code_list, data_type, temp_db)


    leaves_list = create_template_to_leaves(tgt_asts, temp_db,freq = frequent)

    tid2config_seq = product_rules_to_actions_bottomup(tgt_asts, leaves_list, temp_db, rule_type=rule_type,
                                          turn_v_back=turn_v_back, use_normalized_trees=normlize_tree)

    assert isinstance(list(temp_db.action2id.keys())[0], Action), "action2id must contain actions"

    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"


    predicate_freq = {}

    coor_table = dict()

    if train_src_list:
        combine_src_list = src_list + train_src_list
        combine_action_list = tid2config_seq + train_action_seq
        for src_idx, src_split in enumerate(combine_src_list):
            update_predicate_condition_table(src_split, combine_action_list[src_idx], predicate_freq, coor_table)
    else:
        for src_idx, src_split in enumerate(src_list):
            update_predicate_condition_table(src_split, tid2config_seq[src_idx], predicate_freq, coor_table)

    len_of_action_seq = 0
    len_of_code = 0

    for i, src_sent in enumerate(src_list):
        len_of_action_seq += len(tid2config_seq[i])
        len_of_code += len(ori_tgt_code_list[i])
        for action in tid2config_seq[i]:
            predicate_list = get_action_predicate(action)
            action.prototype_tokens = get_predicate_tokens(predicate_list)
            if isinstance(action, ReduceAction):
                vertex = action.rule.head
            else:
                vertex = action.vertex
            vertex.prototype_tokens = action.prototype_tokens
            action.string_sim_score = get_string_sim(src_sent, action.prototype_tokens)
            for ent in action.entities:
                if ent in src_sent:
                    index_list = []
                    for idx, token in enumerate(src_sent):
                        if token == ent:
                            index_list.append(idx + 1)
                    action.entity_align.append(index_list)
                else:
                    action.entity_align.append(-1)
            assert len(action.entity_align) == len(
                action.entities), "align must have the same length with the action entities"
            action.cond_score = get_cond_score(src_sent, action, predicate_freq, coor_table)

        example.append(
            Example(src_sent=src_sent, tgt_code=ori_tgt_code_list[i], tgt_ast=tgt_asts[i],
                    tgt_actions=tid2config_seq[i],
                    idx=i, meta=data_type))
        assert len(tid2config_seq[i]) == len(
            example[i].tgt_ast_seq), "the node head length must be equal to the action length"

    print ("avg action length is {}".format(len_of_action_seq/len(example)))
    print ("avg code length is {}".format(len_of_code/len(example)))
    return example, temp_db

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]