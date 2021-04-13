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

from preprocess_data.nlp_utils import turn_stem_into_lemma
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
from grammar.rule import get_reduce_action_length_set
import nltk
from nltk.stem import WordNetLemmatizer

normalized_levenshtein = NormalizedLevenshtein()


def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_align(data_set, vocab, src_embed, predicate_embed, use_cuda=True, threshold=0.4, map_dict=None, domain_token="flight"):
    c_sum = 0
    c_gen = 0
    c_re = 0
    gen_num = 0
    re_num = 0
    instance_num = 0
    stop_words = set(
        ["is", "the", "are", "about", "doe", "me", "that"])
    for e in data_set:
        print("============================")
        batch = Batch([e], vocab, use_cuda=use_cuda, data_type='lambda')
        if map_dict:
            sent = turn_stem_into_lemma(e.src_sent, map_dict)
        else:
            sent = e.src_sent


        stop_sent = []
        id2id = {}
        for id, word in enumerate(sent):
            if not word in stop_words:
                stop_sent.append(word)
                id2id[len(stop_sent) - 1] = id
        print(batch.src_sents[0])
        src_sents_var = to_input_variable([stop_sent], vocab.source, use_cuda,
                                          append_boundary_sym=False)
        src_embedding = src_embed(src_sents_var)
        src_embedding = src_embedding.squeeze()

        for action in e.tgt_actions:
            #print(action.prototype_tokens)
            align_tokens = set()
            align_ids = set()
            temp_scores = []
            temp_indices = []
            for align_token in [domain_token]:
                if align_token == 'next':
                    print("dasdadsad")
                action_var = to_input_variable([[align_token]], vocab.predicate_tokens_vocab, use_cuda,
                                               append_boundary_sym=False)
                action_embeddings = predicate_embed(action_var)
                action_embedding = action_embeddings.mean(dim=0)
                # print (src_embedding.size())
                # print (action_embedding.unsqueeze(0).size())
                if action_embedding.dim() == 3:
                    action_embedding = action_embedding.squeeze(0)

                if src_embedding.dim() == 1:
                    src_embedding = src_embedding.unsqueeze(0)
                #print (action_embedding.size())
                #print (src_embedding.size())
                distance = torch.cosine_similarity(src_embedding, action_embedding)
                align = distance > threshold
                """
                if isinstance(action, GenAction):
                    if align.sum().item() == 0:
                        tmp_score, index = distance.max(0)
                        temp_scores.append(tmp_score.item())
                        temp_indices.append(index.item())
                """
                for i in range(align.size()[0]):
                    if align[i].item():
                        align_tokens.add(batch.src_sents[0][id2id[i]])
                        align_ids.add(id2id[i] + 1)
            """
            if isinstance(action, GenAction):
                gen_num += 1
                if len(align_tokens) == 0:
                    max_index = temp_indices[temp_scores.index(max(temp_scores))]
                    align_tokens.add(batch.src_sents[0][id2id[max_index]])
                    align_ids.add(id2id[max_index] + 1)
                else:
                    c_gen += 1

            for entity in action.entities:
                entity_splitted = entity.split(':')[0].split('_')
                for ent in entity_splitted:
                    if ent in batch.src_sents[0]:
                        align_ids.add(batch.src_sents[0].index(ent) + 1)
                        align_tokens.add(ent)
                    else:
                        print("{} is not in the sentence ", ent)
            """
            action.align_ids = list(align_ids)
            action.align_tokens = list(align_tokens)
            if isinstance(action, ReduceAction):
                re_num += 1
            if len(action.align_tokens) > 0:
                if isinstance(action, ReduceAction):
                    c_re += 1
                c_sum += 1
            print(action)
            print(action.align_tokens)
        instance_num += (len(e.tgt_actions) - 1)
        print("============================")
    #print("GenAction coverage : {:.4f}".format(c_gen / gen_num))
    #print("ReAction coverage : {:.4f}".format(c_re / re_num))
    #print("Action coverage : {:.4f}".format(c_sum / instance_num))


def prepare_align_actions(train_set, support_set, query_set, glove_path='embedding/glove/glove.6B.200d.txt',
                          embed_size=200, use_cuda=True, threshold=0.4, map_dict=None, domain_token="flight"):
    if query_set is None:
        data_set = train_set + support_set
    else:
        data_set = train_set + support_set + query_set
    if map_dict:
        src_vocab = TokenVocabEntry.from_corpus([turn_stem_into_lemma(e.src_sent, map_dict) for e in data_set],
                                                size=5000, freq_cutoff=0)
    else:
        src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in data_set], size=5000, freq_cutoff=0)
    predicate_tokens_vocab = TokenVocabEntry.from_corpus(
        [action.prototype_tokens + ['geography'] for e in data_set for action in e.tgt_actions], size=5000, freq_cutoff=0)

    src_embed = nn.Embedding(len(src_vocab), embed_size)
    predicate_embed = nn.Embedding(len(predicate_tokens_vocab), embed_size)
    if use_cuda:
        src_embed = src_embed.cuda()
        predicate_embed = predicate_embed.cuda()

    src_glove_embedding = GloveHelper(glove_path, embed_size)
    src_glove_embedding.load_to(src_embed, src_vocab)
    # print (src_embed.weight[12].data)
    predicate_glove_embedding = GloveHelper(glove_path, embed_size)
    predicate_glove_embedding.load_to(predicate_embed, predicate_tokens_vocab)
    vocab = Vocab(source=src_vocab, predicate_tokens_vocab=predicate_tokens_vocab)
    generate_align(data_set, vocab, src_embed, predicate_embed, use_cuda=True, threshold=threshold, map_dict=map_dict, domain_token=domain_token)


def pre_vocab(data_set, temp_db):
    vocab_freq_cutoff = 0
    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in data_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in data_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    entity_vocab = LitEntry.from_corpus(temp_db.entity_list, size=5000, freq_cutoff=0)

    variable_vocab = LitEntry.from_corpus(temp_db.variable_list, size=5000, freq_cutoff=0)

    action_vocab = ActionVocabEntry.from_corpus([e.tgt_actions for e in data_set], size=5000, freq_cutoff=0)
    general_action_vocab = GeneralActionVocabEntry.from_action_vocab(action_vocab.token2id)
    gen_vocab = GenVocabEntry.from_corpus([e.tgt_actions for e in data_set], size=5000, freq_cutoff=0)
    reduce_vocab = ReVocabEntry.from_corpus([e.tgt_actions for e in data_set], size=5000, freq_cutoff=0)
    vertex_vocab = VertexVocabEntry.from_example_list(data_set)
    vocab = Vocab(source=src_vocab, code=code_vocab, action=action_vocab, general_action=general_action_vocab,
                  gen_action=gen_vocab, re_action=reduce_vocab, vertex=vertex_vocab, entity=entity_vocab,
                  variable=variable_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)
    return vocab


def pre_few_shot_vocab(data_set, entity_list, variable_list, disjoint_set, place_holder=None):

    vocab_freq_cutoff = 0
    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in data_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # print ()
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

    """
    prototye_tokens = [action.prototype_tokens for e in data_set for action in e.tgt_actions]
    prototye_tokens.append(['<gen_pad>', '<s>'])
    predicate_tokens_vocab = TokenVocabEntry.from_corpus(prototye_tokens, size=5000, freq_cutoff=0)

    for action in action_vocab.token2id.keys():
        action_tokens = action.prototype_tokens
        if len(action_tokens) == 0:
            predicate_tokens = get_action_predicate(action)
            assert len(predicate_tokens) > 0
            for predicate_token in predicate_tokens:
                src_vocab.add(predicate_token)
        elif 'hidden' in action.prototype_tokens:
            src_vocab.add("and")
            src_vocab.add(str(action.rule.body_length))
        else:
            for token in action_tokens:
                src_vocab.add(token)
    src_vocab.add('predicatetype')
    for i in range(0,5):
        src_vocab.add(str(i))
    """
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
            if child.head == SLOT_PREFIX or child.head == VAR_NAME:
                continue
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


def get_predicate_tokens(predicate_list, use_white_list=True):
    turned_predicate_tokens = []
    for predicate in predicate_list:
        predicate_tokens = predicate.strip().split(':')[0].split('_')
        for token in predicate_tokens:
            #print (token)
            #if token == 'req':
                #print ("key")
            if use_white_list:
                if token in PREDICATE_LEXICON:
                    turned_predicate_tokens.extend(PREDICATE_LEXICON[token])
                else:
                    turned_predicate_tokens.append(token)
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

def parse_overnight_query_helper(elem_list):
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    i = 0
    current = root
    node_pos = 1
    for elem in elem_list:
        if elem == ')':
            current = current.parent
        elif not elem == '(':
            last_elem = elem_list[i - 1]
            if last_elem == '(':
                child = RuleVertex(elem)
                child.parent = current
                current.add(child)
                current = child
                child.position = node_pos
                node_pos += 1
            else:
                child = RuleVertex(elem)
                child.parent = current
                current.add(child)
                child.position = node_pos
                node_pos += 1
        elif elem == '(' and elem_list[i - 1] == '(':
            child = RuleVertex(IMPLICIT_HEAD)
            child.is_auto_nt = True
            child.parent = current
            current.add(child)
            current = child
            child.position = node_pos
            node_pos += 1
        i += 1

    return root


def read_grammar(grammar_dict, data_type):
    path_prefix = "../../datasets/overnight/grammar"
    grammar_path = data_type + ".grammar"
    print (data_type)
    full_path = os.path.join(path_prefix, grammar_path)
    with open(full_path) as grammar_file:
        for line in grammar_file:
            if line.strip():
                line_split = line.strip().split('\t')
                type = ''.join([i for i in line_split[0] if not i.isdigit()])

                rhs = line_split[1]
                if rhs not in grammar_dict:
                    grammar_dict[rhs] = type



def replace_overnight_query(root, data_type):
    grammar_dict = dict()
    read_grammar(grammar_dict, data_type)
    visited, queue = set(), [root]


    while queue:
        vertex = queue.pop(0)
        v_id = id(vertex)
        visited.add(v_id)
        idx = 0
        for child in vertex.children:
            if id(child) not in visited:
                if child.to_lambda_expr in grammar_dict:
                    parent_node = vertex
                    type_vertex = RuleVertex(grammar_dict[child.to_lambda_expr])
                    type_vertex.original_var = child
                    child.is_grammar_vertex = True
                    parent_node.children[idx] = type_vertex
                    type_vertex.parent = parent_node
                else:
                    queue.append(child)
            idx += 1
    return root

def parse_overnight_query(elem_list,data_type):
    root = parse_overnight_query_helper(elem_list)
    root = replace_overnight_query(root, data_type)
    return root

def parse_prolog_query_helper(elem_list, data_type):
    # ( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.filter ( var s ) ( string num_blocks ) ( string ! = ) ( number 3 block ) ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )	overnight-basketball
    # elem_list = ['job', '(', 'ANS', ')', ',', 'language', '(', 'ANS', ',', 'languageid0', ')', ',', '\\+', 'rex_exp', '(', 'area', '(', 'ANS', ',', 'areaid0', ')', ',', 'req_exp', '(', 'ANS', ')', ')', ';', '(', 'area', '(', 'ANS', ',', 'areaid1', ')', ',', 'req_exp', '(', 'ANS', ')', ')']
    separator_set = set([',', '(', ')', ';', '\+'])
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    depth = 0
    i = 0
    current = root
    node_pos = 1
    var_id = 0
    plus_flag = True
    for elem in elem_list:
        # print("{} : {} ".format(elem, current.head))
        if elem == '(':
            depth += 1
            if i > 0:
                last_elem = elem_list[i - 1]
                # if last_elem == '\+':
                # plus_flag = False

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
            # plus_flag = True
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
                # + "_" + str(var_id)
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


def update_predicate_condition_table(src_list, tgt_list, predi_freq, coor_table, data_type):
    for token in tgt_list:
        if token in coor_table:
            for src_token in src_list:
                if src_token in predi_freq:
                    predi_freq[src_token] = predi_freq[src_token] + 1
                else:
                    predi_freq[src_token] = 1

                if src_token in coor_table[token]:
                    coor_table[token][src_token] = coor_table[token][src_token] + 1
                else:
                    coor_table[token][src_token] = 1
        else:
            coor_table[token] = {}
            for src_token in src_list:

                if src_token in predi_freq:
                    predi_freq[src_token] = predi_freq[src_token] + 1
                else:
                    predi_freq[src_token] = 1

                if src_token in coor_table[token]:
                    coor_table[token][src_token] = coor_table[token][src_token] + 1
                else:
                    coor_table[token][src_token] = 1


def get_cond_score(src_sent, predicate, predicate_freq, coor_table):
    cond_score = []
    for token in src_sent:
        cond_score.append(coor_table[predicate][token] / predicate_freq[token])

    cond_score = [0] + cond_score + [0]

    return cond_score


def get_string_sim(src_sent, prototype_tokens):
    sim_score = []
    ignore_tokens = ['A','e','var']
    for src_token in src_sent:
        score_list = [0]
        for tgt_token in prototype_tokens:
            if tgt_token in ignore_tokens:
                score_list.append(0)
            else:
                score_list.append(1 - normalized_levenshtein.distance(src_token, tgt_token))

        sim_score.append(max(score_list))
        # print (src_token)
        # print (prototype_tokens)
        # print (score)
    sim_score = [0] + sim_score + [0]

    return sim_score

def lemmatize(str_list, nltk_lemmer):
    lemma_list = []
    for token in str_list:
        if token in STEM_LEXICON:
            lemma_token = STEM_LEXICON[token]
        else:
            lemma_token = nltk_lemmer.lemmatize(token)
        lemma_list.append(lemma_token)
    return lemma_list

def produce_data(data_filepath, data_type, lang, turn_v_back=False, normlize_tree=True, rule_type="ProductionRuleBLB", parse_mode='bottomup', previous_src_list = None, previous_action_seq = None, frequent = 0, use_white_list = True):
    example = []
    tgt_code_list = []
    src_list = []
    ori_tgt_code_list = []
    nltk_lemmer = WordNetLemmatizer()
    with open(data_filepath) as json_file:
        for line in json_file:
            line_list = line.split('\t')
            src = line_list[0]
            tgt = line_list[1]
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            src = src.strip()
            # src = re.sub(r"((?:c|s|m|r|co|n)\d)", VAR_NAME, src)
            src_split = src.split(' ')
            src_list.append(lemmatize(src_split, nltk_lemmer))
            tgt_split = tgt.split(' ')
            ori_tgt_code_list.append(tgt_split)
            if len(tgt_split) == 1:
                tgt_split = ["(", PAD_PREDICATE] + tgt_split + [")"]
            tgt_code_list.append(tgt_split)
            # calculate the cooccurance
    json_file.close()
    if lang == 'lambda':
        tgt_asts = [parse_lambda_query(t, data_type) for t in tgt_code_list]
    elif lang == 'prolog':
        tgt_asts = [parse_prolog_query(t, data_type) for t in tgt_code_list]
    elif lang == 'overnight':
        tgt_asts = [parse_overnight_query(t, data_type) for t in tgt_code_list]

    temp_db = normalize_trees(tgt_asts)
    get_entity_variable_list(tgt_code_list, data_type, temp_db)


    leaves_list = create_template_to_leaves(tgt_asts, temp_db,freq = frequent)

    tid2config_seq = product_rules_to_actions_bottomup(tgt_asts, leaves_list, temp_db, rule_type=rule_type,
                                          turn_v_back=turn_v_back, use_normalized_trees=normlize_tree)

    assert isinstance(list(temp_db.action2id.keys())[0], Action), "action2id must contain actions"

    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"


    predicate_freq = {}

    coor_table = dict()

    if previous_src_list:
        combine_src_list = src_list + previous_src_list
        combine_action_list = tid2config_seq + previous_action_seq
        for src_idx, src_split in enumerate(combine_src_list):
            update_predicate_condition_table(src_split, combine_action_list[src_idx], predicate_freq, coor_table, data_type)
    else:
        for src_idx, src_split in enumerate(src_list):
            update_predicate_condition_table(src_split, tid2config_seq[src_idx], predicate_freq, coor_table, data_type)

    len_of_action_seq = 0
    len_of_code = 0

    for i, src_sent in enumerate(src_list):
        # todo change it back
        # temp_list = [type(action).__name__ if isinstance(action, ReduceAction) else action for action in tid2config_seq[i]]
        # if i == 52:
        # print ("dsadadsa")
        len_of_action_seq += len(tid2config_seq[i])
        len_of_code += len(ori_tgt_code_list[i])

        for action in tid2config_seq[i]:

            if data_type == 'geo_lambda' or data_type == 'atis_lambda' or data_type == 'job_prolog':
                #print (action)
                predicate_list = get_action_predicate(action)
                action.prototype_tokens = get_predicate_tokens(predicate_list, use_white_list)
                if isinstance(action, ReduceAction):
                    vertex = action.rule.head
                else:
                    vertex = action.vertex
                vertex.prototype_tokens = action.prototype_tokens
                string_sim = get_string_sim(src_sent, action.prototype_tokens)

                # normalize
                #if not (sum(string_sim)) == 0:
                    #string_sim = [float(i)/sum(string_sim) for i in string_sim]

                action.string_sim_score = string_sim
                for ent in action.entities:
                    if ent in src_sent:
                        index_list = []
                        for idx, token in enumerate(src_sent):
                            if token == ent:
                                index_list.append(idx+1)
                        action.entity_align.append(index_list)
                    else:
                        #print (ent)
                        action.entity_align.append(-1)
                assert len(action.entity_align) == len(action.entities), "align must have the same length with the action entities"

            cond_score = get_cond_score(src_sent, action, predicate_freq, coor_table)
            # normalize
            #cond_score = [float(i)/sum(cond_score) for i in cond_score]

            action.cond_score = cond_score

        example.append(
            Example(src_sent=src_sent, tgt_code=ori_tgt_code_list[i], tgt_ast=tgt_asts[i],
                    tgt_actions=tid2config_seq[i],
                    idx=i, meta=data_type))
        if data_type == 'geo_lambda' or data_type == 'atis_lambda' or data_type == 'job_prolog':
            assert len(tid2config_seq[i]) == len(
                example[i].tgt_ast_seq), "the node head length must be equal to the action length"
    # example.sort(key=lambda x : len(x.tgt_actions))
    print ("action length is {}".format(len_of_action_seq/len(example)))
    print ("code length is {}".format(len_of_code/len(example)))
    return example, temp_db

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]