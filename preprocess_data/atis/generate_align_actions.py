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

normlize_tree = True
# frequency
frequency=0
frequency_s=0

supvised = True
use_white_list=False

if normlize_tree == True:
    if frequency>0:
        rule_type = "ProductionRuleBL"
    else:
        rule_type = "ProductionRuleBLB"
else:
    rule_type = "ProductionRuleBL"

map_path_prefix = "datasets/atis/"
generate_dir(map_path_prefix)
glove_path = '../../embedding/glove/glove.6B.300d.txt'

def prepare_geo_align_data(train_set,train_db, support_file, query_file, dump_path_prefix, data_type='atis_lambda', lang='lambda'):
    train_src_list = [e.src_sent for e in train_set]
    train_action_list = [e.tgt_actions for e in train_set]

    c = 0
    for e in train_set:
        if len(e.tgt_actions) == 2:
            c+=1
            continue
        for action in e.tgt_actions:
            if len(action.prototype_tokens) > 1:
                c += 1
                break
    print(c)

    support_set, support_db = produce_data(support_file, data_type, lang,rule_type=rule_type,normlize_tree=normlize_tree,previous_src_list=train_src_list, previous_action_seq=train_action_list,frequent=frequency_s, use_white_list=use_white_list)
    query_set, query_db = produce_data(query_file, data_type, lang,rule_type=rule_type,normlize_tree=normlize_tree, use_white_list=use_white_list)
    """
    prepare_align_actions(train_set, support_set, query_set,
                          glove_path, 300, True, map_dict=None, threshold=0.4, domain_token="flight")
    """

    if supvised:
        train_vocab_set =train_set
        place_holder = 0
    else:
        train_vocab_set = train_set
        place_holder = 6

    train_vocab = pre_few_shot_vocab(data_set=train_vocab_set,
                                     entity_list=train_db.entity_list + support_db.entity_list,
                                     variable_list=train_db.variable_list + support_db.variable_list,
                                     disjoint_set=train_set,place_holder=place_holder)
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

    return support_set, query_set

if __name__ == '__main__':
    train_path_prefix = "../../datasets/atis/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1"
    train_file = os.path.join(train_path_prefix, 'train.txt')
    train_set, train_db = produce_data(train_file, data_type='atis_lambda', lang='lambda',rule_type=rule_type,normlize_tree=normlize_tree,frequent=frequency, use_white_list=use_white_list)

    template_set = set()
    for e in train_set:
        template_set.add(e.to_lambda_template)

    print(train_file)

    support_query_scent = []
    for s in range(6):

        for k in range(2):
            shuffle = s
            shot = k + 1
            path_prefix = "../../datasets/atis/query_split/few_shot_split_random_5_predi/shuffle_{}_shot_{}".format(shuffle, shot)
            if normlize_tree:
                if frequency > 0:
                    dump_path_prefix =  "../../datasets/atis/query_split/freq_{}/few_shot_split_random_5_predi/shuffle_{}_shot_{}".format(frequency, shuffle, shot)
                    print (dump_path_prefix)
                else:
                    dump_path_prefix = "../../datasets/atis/query_split/few_shot_split_random_5_predi/shuffle_{}_shot_{}".format(shuffle, shot)
            else:
                dump_path_prefix = "../../datasets/atis/query_split/unnorm/few_shot_split_random_5_predi/shuffle_{}_shot_{}".format(
                    shuffle, shot)

            if supvised:
                if use_white_list:
                    dump_path_prefix = "../../datasets/atis/query_split/supervised/few_shot_split_random_5_predi/shuffle_{}_shot_{}".format(
                        shuffle, shot)
                else:
                    dump_path_prefix = "../../datasets/atis/query_split/supervised/no_use_white_list/few_shot_split_random_5_predi/shuffle_{}_shot_{}".format(
                        shuffle, shot)

            generate_dir(dump_path_prefix)
            support_file = os.path.join(path_prefix, 'support.txt')
            print (support_file)
            query_file = os.path.join(path_prefix, 'query.txt')
            print (query_file)
            support_set, query_set = prepare_geo_align_data(train_set, train_db, support_file, query_file, dump_path_prefix, data_type='atis_lambda', lang='lambda')
            if s == 0 and k == 0:
                support_query_scent.extend([" ".join(e.src_sent) for e in support_set])
                support_query_scent.extend([" ".join(e.src_sent) for e in query_set])
            else:
                temp_support_query_scent = []
                temp_support_query_scent.extend([" ".join(e.src_sent) for e in support_set])
                temp_support_query_scent.extend([" ".join(e.src_sent) for e in query_set])
                print (len(support_query_scent))
                print (len(temp_support_query_scent))
                for item in support_query_scent:
                    temp_support_query_scent.remove(item)
                print (len(temp_support_query_scent))
                assert len(temp_support_query_scent) == 0, "s {} and k {}".format(s, k)
    pass
