import json
from nltk.corpus import stopwords
from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit
from grammar.action import ReduceAction, GenAction
from grammar.db import create_template_to_leaves
from grammar.rule import Action
from preprocess_data.atis.generate_actions import produce_data
from components.dataset import Example
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry
import os
import sys
from itertools import chain
import re

from preprocess_data.data_generation_utils import query_split_data

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from grammar.rule import get_reduce_action_length_set

RATIO = 0.5
overlap_predicate = 10
path_prefix = "../../datasets/atis/"
dump_path_prefix = "../../datasets/atis/query_split_previous_5/few_shot_split_random_{}_predi".format(overlap_predicate)



def split_by_new_token(query_dict, ratio = RATIO, num_non_overlap = 10, test_oov=True):
    query_dict_list = list(query_dict.items())
    np.random.shuffle(query_dict_list)
    oov_train_dict = {}
    test_oov_dict = {}
    test_no_oov_dict = {}
    vocab_set = set()
    train_dict = dict(query_dict_list[int(len(query_dict) * ratio):])
    test_dict = dict(query_dict_list[:int(len(query_dict) * ratio)])
    for k, v in test_dict.items():
        for word in k.split('|||'):
            vocab_set.add(word)
    vocab_list = list(vocab_set)
    vocab_list = vocab_list[:num_non_overlap]
    print("vocab list ", vocab_list)
    for k, v in train_dict.items():
        flag = True
        for word in k.split('|||'):
            if word in vocab_list:
                flag = False
                break
        if flag:
            oov_train_dict[k] = v

    for k, v in test_dict.items():
        flag = True
        for word in k.split('|||'):
            if word in vocab_list:
                test_oov_dict[k] = v
                flag = False
                break
        if flag:
            test_no_oov_dict[k] = v
    if test_oov == True:
        return oov_train_dict, test_oov_dict
    else:
        return oov_train_dict, test_no_oov_dict


def split_by_ratio(query_dict, ratio=0.5):
    return dict(list(query_dict.items())[int(len(query_dict) * ratio):]), dict(
        list(query_dict.items())[:int(len(query_dict) * ratio)])

def query_split(data_set, split_mode="ratio"):
    query_dict = {}
    for e in data_set:
        template = "|||".join([str(action) for action in e.tgt_actions])
        if template in query_dict:
            query_dict[template].append(e)
        else:
            query_dict[template] = []
            query_dict[template].append(e)

    if split_mode == 'ratio':
        train, test = split_by_ratio(query_dict, ratio=RATIO)
    elif split_mode == 'oov_split':
        train, test = split_by_new_token(query_dict, ratio=RATIO, test_oov=True)
    return train, test

def prepare_atis_lambda(train_file, dev_file, test_file):
    vocab_freq_cutoff = 0
    train_set, train_action2id = produce_data(train_file)
    dev_set, dev_action2id = produce_data(dev_file)
    test_set, test_action2id = produce_data(test_file)

    train_set, test_set = query_split(train_set+dev_set+ test_set,split_mode='oov_split')

    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)
    action_vocab = ActionVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    general_action_vocab = GeneralActionVocabEntry.from_action_vocab(action_vocab.token2id)
    gen_vocab = GenVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    reduce_vocab = ReVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    vertex_vocab = VertexVocabEntry.from_example_list(train_set)
    vocab = Vocab(source=src_vocab, code=code_vocab, action=action_vocab, general_action=general_action_vocab,
                  gen_action=gen_vocab, re_action=reduce_vocab, vertex = vertex_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, test_set)]
    print('Train set len: %d' % len(train_set))
    print('Test set len: %d' % len(test_set))
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))


if __name__ == '__main__':
    overlap_predicate = 5
    predicate_list = ["departure_time", "has_meal", "class_type", "meal", "ground_fare"]
    path_prefix = "../../datasets/atis/"
    dump_path_prefix = "../../datasets/atis/query_split_previous_5/few_shot_split_random_{}_predi/".format(overlap_predicate)
    train_file = os.path.join(path_prefix, 'train.txt')
    dev_file = os.path.join(path_prefix, 'dev.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    query_split_data(train_file, dev_file, test_file, dump_path_prefix, lang = 'lambda', data_type = 'atis_lambda', num_non_overlap = overlap_predicate, predicate_list=predicate_list)
    pass
