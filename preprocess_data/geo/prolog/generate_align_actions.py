import os
import pickle
import sys
from itertools import chain
import sys
sys.path.append('./')
import numpy as np

from model.utils import GloveHelper
from preprocess_data.utils import produce_data, get_predicate_tokens, prepare_align_actions, pre_vocab, generate_dir, \
    pre_few_shot_vocab
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
import torch.nn as nn
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry, \
    LitEntry

map_path_prefix = "datasets/geo/"
generate_dir(map_path_prefix)
glove_path = 'embedding/glove/glove.6B.300d.txt'


def prepare_geo_align_data(train_file, support_file, query_file, dump_path_prefix, data_type='atis_lambda', lang='lambda'):

    train_set, train_db = produce_data(train_file, data_type, lang)
    support_set, support_db = produce_data(support_file, data_type, lang)
    query_set, query_db = produce_data(query_file, data_type, lang)

    train_vocab = pre_few_shot_vocab(data_set=train_set + support_set,
                                     entity_list=train_db.entity_list + support_db.entity_list,
                                     variable_list=train_db.variable_list + support_db.variable_list,
                                     disjoint_set=train_set)
    support_vocab = pre_few_shot_vocab(data_set=train_set + support_set,
                                       entity_list=train_db.entity_list + support_db.entity_list,
                                       variable_list=train_db.variable_list + support_db.variable_list,
                                       disjoint_set=support_set)
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


if __name__ == '__main__':

    for s in range(5):
        for k in range(2):
            shuffle = s
            shot = k + 1
            path_prefix = "../../datasets/jobs/query_split_previous_5/few_shot_split_random_3_predi/shuffle_{}_shot_{}".format(shuffle, shot)
            train_file = os.path.join(path_prefix, 'train.txt')
            print (train_file)
            support_file = os.path.join(path_prefix, 'support.txt')
            print (support_file)
            query_file = os.path.join(path_prefix, 'query.txt')
            print (query_file)
            prepare_geo_align_data(train_file, support_file, query_file, path_prefix, data_type='atis_lambda', lang='lambda')
    pass