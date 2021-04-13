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

frequency=0

normlize_tree = True
if normlize_tree:
    if frequency > 0:
        rule_type = "ProductionRuleBL"
    else:
        rule_type = "ProductionRuleBLB"
else:
    rule_type = "ProductionRuleBL"

map_path_prefix = "datasets/jobs/"
generate_dir(map_path_prefix)
glove_path = 'embedding/glove/glove.6B.300d.txt'

place_holder = 0

def prepare_geo_align_data(train_file, support_file, query_file, dump_path_prefix, data_type='atis_lambda', lang='lambda'):

    train_set, train_db = produce_data(train_file, data_type, lang,rule_type=rule_type,normlize_tree=normlize_tree,frequent=frequency)

    template_set = set()
    for e in train_set:
        template_set.add(e.to_lambda_template)


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

    train_src_list = [e.src_sent for e in train_set]
    train_action_list = [e.tgt_actions for e in train_set]

    support_set, support_db = produce_data(support_file, data_type, lang,rule_type=rule_type,normlize_tree=normlize_tree, previous_src_list=train_src_list, previous_action_seq=train_action_list)
    query_set, query_db = produce_data(query_file, data_type, lang,rule_type=rule_type,normlize_tree=normlize_tree)
    """
    prepare_align_actions(train_set, support_set, query_set,
                          glove_path, embed_size, use_cuda, map_dict=map_dict_obj, threshold=thres)
    """


    train_vocab = pre_few_shot_vocab(data_set=train_set,
                                     entity_list=train_db.entity_list + support_db.entity_list,
                                     variable_list=train_db.variable_list + support_db.variable_list,
                                     disjoint_set=train_set, place_holder=place_holder)
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


    support_query_scent = []
    for s in range(5):
        for k in range(2):
            shuffle = s
            shot = k + 1
            path_prefix = "../../datasets/jobs/query_split/few_shot_split_random_3_predishuffle_{}_shot_{}".format(shuffle, shot)

            dump_path_prefix = "../../datasets/jobs/query_split/ratsql_freq_{}/few_shot_split_random_3_predi/shuffle_{}_shot_{}".format(frequency,
                    shuffle, shot)


            generate_dir(dump_path_prefix)
            train_file = os.path.join(path_prefix, 'train.txt')
            print (train_file)
            support_file = os.path.join(path_prefix, 'support.txt')
            print (support_file)
            query_file = os.path.join(path_prefix, 'query.txt')
            print (query_file)
            support_set, query_set = prepare_geo_align_data(train_file, support_file, query_file, dump_path_prefix, data_type='job_prolog', lang='prolog')
            if s == 0 and k == 0:
                support_query_scent.extend([" ".join(e.src_sent) for e in support_set])
                support_query_scent.extend([" ".join(e.src_sent) for e in query_set])
            else:
                temp_support_query_scent = []
                temp_support_query_scent.extend([" ".join(e.src_sent) for e in support_set])
                temp_support_query_scent.extend([" ".join(e.src_sent) for e in query_set])
                print(len(support_query_scent))
                print(len(temp_support_query_scent))
                for item in support_query_scent:
                    temp_support_query_scent.remove(item)
                print(len(temp_support_query_scent))
                assert len(temp_support_query_scent) == 0, "s {} and k {}".format(s, k)
    pass
