import json
from nltk.corpus import stopwords
from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit, is_predicate
from grammar.action import ReduceAction, GenAction
from grammar.db import create_template_to_leaves
from grammar.rule import Action
from components.dataset import Example
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry, LitEntry
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
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
from preprocess_data.utils import parse_prolog_query_helper, produce_data

data_format = 'job_prolog'


path_prefix = "../../datasets/jobs/"
dump_path_prefix = "../../datasets/jobs/"

def prepare_jobs_prolog(train_file, test_file):
    vocab_freq_cutoff = 0
    train_set, train_temp_db = produce_data(train_file, 'job_prolog', 'prolog', rule_type="ProductionRuleBLB")
    test_set, test_temp_db = produce_data(test_file, 'job_prolog', 'prolog', rule_type="ProductionRuleBLB")
    predicate_set = set()
    predicate_freq = dict()
    for e in train_set:
        for token in e.tgt_code:
            if is_predicate(token, 'job_prolog'):
                predicate_set.add(token)
                if token in predicate_freq:
                    predicate_freq[token] = predicate_freq[token] + 1
                else:
                    predicate_freq[token] = 1
    for e in test_set:
        for token in e.tgt_code:
            if is_predicate(token, 'job_prolog'):
                predicate_set.add(token)
                if token in predicate_freq:
                    predicate_freq[token] = predicate_freq[token] + 1
                else:
                    predicate_freq[token] = 1
    print (len(predicate_set))
    print (predicate_freq)
    print ({k: v for k, v in sorted(predicate_freq.items(), key=lambda item: item[1])})
    print (sum(predicate_freq.values()))

    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    entity_vocab = LitEntry.from_corpus(train_temp_db.entity_list, size=5000, freq_cutoff=0)

    variable_vocab = LitEntry.from_corpus(train_temp_db.variable_list, size=5000, freq_cutoff=0)

    action_vocab = ActionVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)

    src_list = [e.src_sent for e in train_set]
    src_list += [["and", str(action.rule.body_length)] if 'hidden' in action.prototype_tokens else action.prototype_tokens for action in action_vocab.token2id.keys()]
    src_vocab = TokenVocabEntry.from_corpus(src_list, size=5000, freq_cutoff=vocab_freq_cutoff)

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
    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))

if __name__ == '__main__':
    #for s in range(5):
        #for k in range(2):
            #path_prefix = "../../datasets/jobs/query_split/supervised/few_shot_split_random_3_predi/shuffle_{}_shot_{}".format(s, k+1)
            #dump_path_prefix = "../../datasets/jobs/query_split/supervised/shuffle_{}_{}".format(s, k+1)
    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    prepare_jobs_prolog(train_file,test_file)
    pass
