import json
from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit
from grammar.action import ReduceAction
from grammar.db import create_template_to_leaves
from grammar.rule import product_rules_to_actions, get_reduce_action_length_set
from grammar.rule import Action
from components.dataset import Example
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry
import os
import sys
from itertools import chain

try: import cPickle as pickle
except: import pickle
import numpy as np

path_prefix = "../../../datasets/geo/lambda/question_split/"
dump_path_prefix = "../../../datasets/geo/lambda/question_split/recursive_lstm/"

def parse_lambda_query_helper(elem_list):
    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    i = 0
    current = root
    node_pos = 1
    # (A, (_major(A), _city(A), _loc(A, B), _state(B), _river(C), _loc(C, D), _const(D, _stateid(var0)), _traverse(C, B)))
    # ( lambda $0 e ( and ( state:t $0 ) ( exists $1 ( and ( place:t $1 ) ( > ( elevation:i $1 ) ( elevation:i ( argmax $2 ( and ( place:t $2 ) ( loc:t $2 s0 ) ) ( elevation:i $2 ) ) ) ) ) ) ) )
    for elem in elem_list:
        if elem == ')':
            current = current.parent
        elif not elem in [',','(',';']:
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
                if is_variable:
                    child.original_var = elem
                child.parent = current
                current.add(child)
                child.position = node_pos
                node_pos += 1
        i+=1

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
            src , tgt = line.split('\t')
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            src = src.strip()
            src_list.append(src.split(' '))
            tgt_code_list.append(tgt.split(' '))


    tgt_asts = [parse_lambda_query(t) for t in tgt_code_list]

    temp_db = normalize_trees(tgt_asts)
    leaves_list = create_template_to_leaves(tgt_asts, temp_db)
    tid2config_seq = product_rules_to_actions(tgt_asts,leaves_list, temp_db, True)
    reduce_action_length_set = get_reduce_action_length_set(tid2config_seq)
    for action, id in temp_db.action2id.items():
        if isinstance(action, ReduceAction):
            action.rule.body_length_set = reduce_action_length_set[action]


    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"
    assert isinstance(list(temp_db.action2id.keys())[0], Action), "action2id must contain actions"
    for i, src_sent in enumerate(src_list):
        # todo change it back
        #temp_list = [type(action).__name__ if isinstance(action, ReduceAction) else action for action in tid2config_seq[i]]
        example.append(Example(src_sent = src_sent,tgt_code = tgt_code_list[i], tgt_ast = tgt_asts[i], tgt_actions = tid2config_seq[i], idx = i, meta = None))
    return example, temp_db.action2id

def prepare_geo_lambda(train_file, test_file):
    vocab_freq_cutoff = 0
    train_set, train_action2id = produce_data(train_file)
    test_set, test_action2id = produce_data(test_file)
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
    print ('Test set len: %d' % len(test_set))
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix,'vocab.freq2.bin'), 'wb'))


if __name__ == '__main__':
    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    prepare_geo_lambda(train_file,test_file)
    pass
