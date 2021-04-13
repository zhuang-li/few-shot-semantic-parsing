import json
from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit
from grammar.db import create_template_to_leaves
from grammar.rule import product_rules_to_actions
from grammar.rule import Action
from components.dataset import Example
from components.vocab import Vocab
from grammar.action import ReduceAction
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry

import sys
from itertools import chain

from preprocess_data.geo.process_geoquery import q_process

try: import cPickle as pickle
except: import pickle
import numpy as np

def parse_prolog_query_helper(bracket_query, var_name = 'A', separator_set = set([',','(',')',';'])):
    elem_list = bracket_query.split(' ')
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    depth = 0
    i = 0
    current = root
    node_pos = 1
    for elem in elem_list:
        #print("{} : {} ".format(elem, current.head))
        if elem == '(':
            depth += 1
            if i > 0:
                last_elem = elem_list[i-1]
                if last_elem in separator_set:
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
        elif not elem == ',':
            is_variable = is_var(elem, dataset='geo_prolog')
            is_literal = is_lit(elem, dataset='geo_prolog')
            if is_variable:
                norm_elem = var_name
            elif is_literal:
                norm_elem = SLOT_PREFIX
            else:
                norm_elem = elem
            child = RuleVertex(norm_elem)
            if is_variable:
                child.original_var = elem
            child.parent = current
            current.add(child)
            child.position = node_pos
            node_pos += 1
        i+=1

    return root


def parse_prolog_query(bracket_query):
    root = parse_prolog_query_helper(bracket_query)
    return root

def produce_data(template_filepath, train = ['train', 'dev'], is_query_split = True):
    example = []
    tgt_code_list = []
    tgt_code_list_splited = []
    src_list = []
    with open(template_filepath) as json_file:
        json_data = json.load(json_file)
        for program in json_data:
            if not is_query_split:
                for q_s in program['sentences']:
                    if q_s['question-split'] in train:
                        template = program['logic'][0]
                        tgt_code_list.append(template)
                        tgt_code_list_splited.append(template.split(' '))
                        for var,name in q_s['variables'].items():
                            q_s['text'] = q_s['text'].replace(var,name)
                        src_list.append(q_process(q_s['text'])[0])
            else:
                if program['query-split'] in train:
                    for q_s in program['sentences']:
                        template = program['logic'][0]
                        tgt_code_list.append(template)
                        tgt_code_list_splited.append(template.split(' '))
                        for var,name in q_s['variables'].items():
                            q_s['text'] = q_s['text'].replace(var,name)
                        src_list.append(q_process(q_s['text'])[0])


    tgt_asts = [parse_prolog_query(t) for t in tgt_code_list]
    temp_db = normalize_trees(tgt_asts)
    leaves_list = create_template_to_leaves(tgt_asts, temp_db)
    tid2config_seq = product_rules_to_actions(tgt_asts,leaves_list, temp_db, True)
    reduce_action_length_set = {}
    for action_seq in tid2config_seq:
        for action in action_seq:
            if isinstance(action, ReduceAction):
                rule_len = action.rule.body_length
                if action not in reduce_action_length_set:
                    reduce_action_length_set[action] = set()
                if rule_len not in reduce_action_length_set[action]:
                    reduce_action_length_set[action].add(rule_len)
    for action, id in temp_db.action2id.items():
        if isinstance(action, ReduceAction):
            action.rule.body_length_set = reduce_action_length_set[action]
    assert len(src_list) == len(tgt_code_list_splited), "instance numbers should be consistent"
    assert isinstance(list(temp_db.action2id.keys())[0], Action), "action2id must contain actions"
    for i, src_sent in enumerate(src_list):
        # todo change it back
        example.append(Example(src_sent = src_sent,tgt_code = tgt_code_list_splited[i], tgt_ast = tgt_asts[i], tgt_actions = tid2config_seq[i], idx = i, meta = None))
    return example, temp_db.action2id

def prepare_geo_prolog(template_filepath, is_query_split):
    vocab_freq_cutoff = 0
    train_set, train_action2id = produce_data(template_filepath, ["train" ,"dev"], is_query_split)
    dev_set, dev_action2id = produce_data(template_filepath, ["dev"], is_query_split)

    test_set, test_action2id = produce_data(template_filepath, ["test"], is_query_split)
    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)
    action_vocab = ActionVocabEntry.from_action2id(action2id=train_action2id)
    general_action_vocab = GeneralActionVocabEntry.from_action_vocab(action_vocab.token2id)
    # todo change it back
    vocab = Vocab(source=src_vocab, code=code_vocab, action = action_vocab, general_action = general_action_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, test_set)]
    print ('Train set len: %d' % len(train_set))
    print ('Dev set len: %d' % len(dev_set))
    print ('Test set len: %d' % len(test_set))
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open('../../../datasets/geo/prolog/question_split/train.bin', 'wb'))
    pickle.dump(dev_set, open('../../../datasets/geo/prolog/question_split/dev.bin', 'wb'))
    pickle.dump(test_set, open('../../../datasets/geo/prolog/question_split/test.bin', 'wb'))
    pickle.dump(vocab, open('../../../datasets/geo/prolog/question_split/vocab.freq2.bin', 'wb'))


if __name__ == '__main__':
    template_filepath = '../../../datasets/geo/prolog/geo_query_split.txt'
    prepare_geo_prolog(template_filepath, False)
    pass
