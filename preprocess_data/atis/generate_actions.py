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
from grammar.rule import get_reduce_action_length_set
from preprocess_data.utils import parse_lambda_query_helper

path_prefix = "../../datasets/atis/atis_aug/shuffle_0_shot_2"
dump_path_prefix = "../../datasets/atis/atis_aug/shuffle_0_shot_2"
generate_dir(path_prefix)
generate_dir(dump_path_prefix)



def prepare_geo_lambda(train_file, dev_file, test_file):
    vocab_freq_cutoff = 0
    train_set, train_temp_db = produce_data(train_file, 'atis_lambda', 'lambda')
    test_set, test_temp_db = produce_data(test_file, 'atis_lambda', 'lambda')
    dev_set, dev_temp_db = produce_data(dev_file, 'atis_lambda', 'lambda')

    dataset = test_set + dev_set
    predicate_set = set()
    predicate_freq = {}
    for e in dataset:
        for token in e.tgt_code:
            if is_predicate(token, 'atis_lambda'):
                predicate_set.add(token)
                if token in predicate_freq:
                    predicate_freq[token] = predicate_freq[token] + 1
                else:
                    predicate_freq[token] = 1
    print (len(predicate_set))
    print (predicate_freq)
    print ({k: v for k, v in sorted(predicate_freq.items(), key=lambda item: item[1])})
    print (sum(predicate_freq.values()))

    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    entity_vocab = LitEntry.from_corpus(train_temp_db.entity_list, size=5000, freq_cutoff=0)

    variable_vocab = LitEntry.from_corpus(train_temp_db.variable_list, size=5000, freq_cutoff=0)

    action_vocab = ActionVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    general_action_vocab = GeneralActionVocabEntry.from_action_vocab(action_vocab.token2id)
    gen_vocab = GenVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    reduce_vocab = ReVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    vertex_vocab = VertexVocabEntry.from_example_list(train_set)
    vocab = Vocab(source=src_vocab, code=code_vocab, action=action_vocab, general_action=general_action_vocab,
                  gen_action=gen_vocab, re_action=reduce_vocab, vertex = vertex_vocab, entity = entity_vocab, variable = variable_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, test_set)]
    print('Train set len: %d' % len(train_set))
    print('Test set len: %d' % len(test_set))
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    pickle.dump(dev_set, open(os.path.join(dump_path_prefix, 'support.bin'), 'wb'))
    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'query.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))



if __name__ == '__main__':
    train_file = os.path.join(path_prefix, 'train.txt')
    dev_file = os.path.join(path_prefix, 'support.txt')
    test_file = os.path.join(path_prefix, 'query.txt')
    prepare_geo_lambda(train_file, dev_file, test_file)
    pass
