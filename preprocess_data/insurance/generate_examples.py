import sys
sys.path.append('./')
from preprocess_data.utils import *
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry
import os
import sys
from itertools import chain
import re

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from preprocess_data.utils import parse_lambda_query_helper


def produce_data(data_filepath):
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
            tgt_code_list.append(tgt_split)

    json_file.close()




    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"

    for i, src_sent in enumerate(src_list):
        example.append(
            Example(src_sent=src_sent, tgt_code=ori_tgt_code_list[i], tgt_ast=[],
                    tgt_actions=[],
                    idx=i, meta=None))
    return example



def prepare_insurance_lambda(train_file, test_file):
    vocab_freq_cutoff = 0
    train_set = produce_data(train_file)

    test_set = produce_data(test_file)


    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    vocab = Vocab(source=src_vocab, code=code_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)


    print('Train set len: %d' % len(train_set))
    print('Test set len: %d' % len(test_set))


    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))



if __name__ == '__main__':
    path_prefix = "../../datasets/insurance/"
    dump_path_prefix = "../../datasets/insurance/"
    generate_dir(path_prefix)
    generate_dir(dump_path_prefix)

    train_file = os.path.join(path_prefix, 'SQL_train.txt')
    test_file = os.path.join(path_prefix, 'SQL_train.txt')
    prepare_insurance_lambda(train_file, test_file)
    pass
