# coding=utf-8
from __future__ import print_function

import os
from collections import Counter

from nltk import WordNetLemmatizer

from grammar.action import *
from grammar.vertex import *
from grammar.rule import *
from itertools import chain

class VocabEntry(object):
    def __getitem__(self, token):
        return self.token2id.get(token, self.unk_id)

    def __contains__(self, token):
        return token in self.token2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.token2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def index(self, token):
        return self[token]

    def indices(self, tokens):
        return [self.index(token) for token in tokens]

    def id2token(self, wid):
        return self.id2token[wid]

    def add(self, token):
        if token not in self:
            wid = self.token2id[token] = len(self)
            self.id2token[wid] = token
            return wid
        else:
            return self[token]

    def is_unk(self, token):
        return token not in self

class LitEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.token2id['<pad>'] = 0
        self.id2token = {v: k for k, v in self.token2id.items()}

    def __getitem__(self, token):
        return self.token2id.get(token)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = LitEntry()

        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if len(vocab_entry) < size:
                if token_freq[token] >= freq_cutoff:
                    vocab_entry.add(token)
                else:
                    tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)

        return vocab_entry

class TokenVocabEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.unk_id = 3
        self.token2id['<pad>'] = 0
        self.token2id['<s>'] = 1
        self.token2id['</s>'] = 2
        self.token2id['<unk>'] = 3
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.top_k_tokens = []


    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0, selected_freq = 50):
        vocab_entry = TokenVocabEntry()
        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if len(vocab_entry) < size:
                if token_freq[token] >= freq_cutoff:
                    vocab_entry.add(token)
                else:
                    tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)
        vocab_entry.top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:selected_freq]

        return vocab_entry

class ActionVocabEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.token2id[GenAction(RuleVertex('<gen_pad>'))] = 0
        self.token2id[GenAction(RuleVertex('<s>'))] = 1
        self.id2token = {v: k for k, v in self.token2id.items()}

    def __getitem__(self, token):
        return self.token2id.get(token)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = ActionVocabEntry()

        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if len(vocab_entry) < size:
                if token_freq[token] >= freq_cutoff:
                    vocab_entry.add(token)
                else:
                    tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)

        return vocab_entry

    @staticmethod
    def from_action2id(action2id):
        vocab_entry = ActionVocabEntry()
        for k, v in action2id.items():

            vocab_entry.add(k)
        return vocab_entry

class GenVocabEntry(VocabEntry):
    def __init__(self):
        self.vocab_type = GenAction.__name__
        self.token2id = dict()
        self.token2id[GenAction(RuleVertex('<gen_pad>'))] = 0
        self.token2id[GenAction(RuleVertex('<s>'))] = 1

        self.id2token = {v: k for k, v in self.token2id.items()}

    def __getitem__(self, token):
        return self.token2id.get(token)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = GenVocabEntry()

        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if type(token).__name__ == vocab_entry.vocab_type:
                if len(vocab_entry) < size:
                    if token_freq[token] >= freq_cutoff:
                        vocab_entry.add(token)
                    else:
                        tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)

        return vocab_entry

    @staticmethod
    def from_action2id(action2id):
        vocab_entry = GenVocabEntry()
        for k, v in action2id.items():
            if type(k).__name__ == vocab_entry.vocab_type:
                vocab_entry.add(k)
        return vocab_entry

class VertexVocabEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.token2id[RuleVertex('<gen_pad>')] = 0
        self.token2id[RuleVertex('<s>')] = 1

        self.id2token = {v: k for k, v in self.token2id.items()}

    def __getitem__(self, token):
        return self.token2id.get(token)

    @staticmethod
    def from_example_list(example_list):
        vocab_entry = VertexVocabEntry()
        for example in example_list:
            for head in example.tgt_ast_seq:
                vocab_entry.add(head)
        return vocab_entry

class ReVocabEntry(VocabEntry):
    def __init__(self):
        self.vocab_type = ReduceAction.__name__
        self.token2id = dict()
        self.token2id[ReduceAction(Rule(RuleVertex('<re_pad>')))] = 0
        self.id2token = {v: k for k, v in self.token2id.items()}

    def __getitem__(self, token):
        return self.token2id.get(token)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = ReVocabEntry()

        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if type(token).__name__ == vocab_entry.vocab_type:
                if len(vocab_entry) < size:
                    if token_freq[token] >= freq_cutoff:
                        vocab_entry.add(token)
                    else:
                        tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)

        return vocab_entry

    @staticmethod
    def from_action2id(action2id):
        vocab_entry = ReVocabEntry()
        for k, v in action2id.items():
            if type(k).__name__ == vocab_entry.vocab_type:
                vocab_entry.add(k)
        #vocab_entry.init_batch_label()
        return vocab_entry

class GeneralActionVocabEntry(VocabEntry):
    def __init__(self):
        self.action_hierarchy = dict()
        self.token2id = dict()

        self.id2token = {v: k for k, v in self.token2id.items()}

    def add_token(self, general_token):
        if general_token not in self:
            wid = self.token2id[general_token] = len(self)
            self.id2token[wid] = general_token
            return wid
        else:
            return self[general_token]

    def __getitem__(self, token):
        general_token = type(token).__name__
        return self.token2id.get(general_token)


    @staticmethod
    def from_action_vocab(action_vocab_dict):
        vocab_entry = GeneralActionVocabEntry()
        for token, id in action_vocab_dict.items():
            general_token = type(token).__name__
            vocab_entry.add_token(general_token)
        return vocab_entry

class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %stokens' % (entry, getattr(self, entry)) for entry in self.entries))

