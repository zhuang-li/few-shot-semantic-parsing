# coding=utf-8
from __future__ import print_function

import os
from collections import Counter

from nltk import WordNetLemmatizer

from grammar.action import *
from grammar.vertex import *
from grammar.consts import STEM_LEXICON
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
        #print (token)
        #print (self[token])
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
        #self.token2id['<s>'] = 1
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
        # self.lemma_token2id = dict()

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
        """
        for token, id in vocab_entry.token2id.items():
            if token in STEM_LEXICON:
                lemma_token = STEM_LEXICON[token]
                vocab_entry.lemma_token2id[lemma_token] = id
            else:
                vocab_entry.lemma_token2id[token] = id
        assert len(vocab_entry.lemma_token2id.items()) == len(vocab_entry.token2id.items()), "lemma token length should be the same with token length"
        """
        return vocab_entry

class ActionVocabEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.token2id[GenAction(RuleVertex('<gen_pad>'))] = 0
        self.token2id[GenAction(RuleVertex('<s>'))] = 1
        #self.token2id[ReduceAction(Rule(RuleVertex('<re_pad>')))] = 2
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



class GenNTVocabEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.token2id[GenNTAction(RuleVertex('<pad>'))] = 0
        self.id2token = {v: k for k, v in self.token2id.items()}

    def __getitem__(self, token):
        return self.token2id.get(token)

    def copy(self):
        vocab_entry = GenNTVocabEntry()

        for action, id in self.token2id.items():
            vocab_entry.token2id[action.copy()] = id

        vocab_entry.id2token = {v: k for k, v in self.token2id.items()}

        return vocab_entry

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = GenNTVocabEntry()

        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if isinstance(token, GenNTAction):
                if len(vocab_entry) < size:
                    if token_freq[token] >= freq_cutoff:
                        vocab_entry.add(token)
                    else:
                        tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)
        for id, action in vocab_entry.id2token.items():
            init_position(action.vertex, 0)
        return vocab_entry


class GenTVocabEntry(VocabEntry):
    def __init__(self):
        self.token2id = dict()
        self.token2id[GenTAction(RuleVertex('<pad>'))] = 0
        self.id2token = {v: k for k, v in self.token2id.items()}

        self.vertex2action = dict()

        self.lhs2rhs = dict()
        self.rhs2lhs = dict()
        self.lhs2rhsid = dict()

        self.entype2action = dict()
        self.action2nl = dict()



    def __getitem__(self, token):
        return self.token2id.get(token)

    def read_grammar(self, file_path, data_type):
        self.vertex2action = {k.vertex.to_lambda_expr:k for k, v in self.token2id.items()}
        print (self.token2id)
        nltk_lemmer = WordNetLemmatizer()
        grammar_path = data_type + ".grammar"
        full_path = os.path.join(file_path, grammar_path)

        with open(full_path) as grammar_file:
            for line in grammar_file:
                if line.strip():
                    line_split = line.strip().split('\t')
                    lhs = ''.join([i for i in line_split[0] if not i.isdigit()])
                    rhs = line_split[1]
                    if lhs == '$EntityNP' and rhs in self.vertex2action:

                        action = self.vertex2action[rhs]
                        action.type = action.get_vertex_type()
                        entype = action.type
                        if action not in self.action2nl:
                            self.action2nl[action] = []
                        nl_set = set()
                        for nl in line_split[2:]:
                            nl_set.add(' '.join([nltk_lemmer.lemmatize(token) for token in nl.split(' ')]))

                        self.action2nl[action] = list(nl_set)

                        if entype in self.entype2action:
                            self.entype2action[entype].append(action)
                        else:
                            self.entype2action[entype] = []
                            self.entype2action[entype].append(action)


                    if rhs not in self.rhs2lhs:
                        self.rhs2lhs[rhs] = lhs
                    if lhs not in self.lhs2rhs:
                        self.lhs2rhs[lhs] = []
                        self.lhs2rhs[lhs].append(rhs)
                        self.lhs2rhsid[lhs] = []

                        action = self.vertex2action[rhs]

                        self.lhs2rhsid[lhs].append(self.token2id[action])
                    else:
                        if rhs in self.vertex2action:
                            action = self.vertex2action[rhs]
                            self.lhs2rhs[lhs].append(rhs)
                            self.lhs2rhsid[lhs].append(self.token2id[action])

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = GenTVocabEntry()

        token_freq = Counter(chain(*corpus))
        non_singletons = [w for w in token_freq if token_freq[w] > 1]
        singletons = [w for w in token_freq if token_freq[w] == 1]
        print('number of token types: %d, number of token types w/ frequency > 1: %d' % (len(token_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_tokens = sorted(token_freq.keys(), reverse=True, key=token_freq.get)[:size]
        tokens_not_included = []
        for token in top_k_tokens:
            if isinstance(token, GenTAction):
                if len(vocab_entry) < size:
                    if token_freq[token] >= freq_cutoff:
                        vocab_entry.add(token)
                    else:
                        tokens_not_included.append(token)

        print('token types not included: %s' % tokens_not_included)
        for id, action in vocab_entry.id2token.items():
            init_position(action.vertex, 0)
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
        self.body_len2token = dict()
        self.body_len2token_batch = []
        self.body_len2token_flatten = []
        self.body_len_list = []
        self.body_len_length_array = []

    def __getitem__(self, token):
        return self.token2id.get(token)

    def init_batch_label(self):
        body_len_dict = dict()
        for token in self.token2id.keys():
            if not token.rule.body_length == 0:
                if token.rule.body_length in body_len_dict:
                    body_len_dict[token.rule.body_length].append(token)
                else:
                    body_len_dict[token.rule.body_length] = []
                    body_len_dict[token.rule.body_length].append(token)
        self.body_len2token = body_len_dict
        body_len_list = sorted(list(body_len_dict.keys()))
        self.body_len_list = body_len_list
        for body_len in body_len_list:
            self.body_len2token_batch.append(body_len_dict[body_len])
            self.body_len2token_flatten.extend(body_len_dict[body_len])
        assert len(self.token2id) - 1 == len(self.body_len2token_flatten), "length of token2id {0} should equal to the length of body_len2token_flatten {1}".format(len(self.token2id), len(self.body_len2token_flatten))
        self.body_len_length_array = [len(e) for e in self.body_len2token_batch]
        for id, action in enumerate(self.body_len2token_flatten):
            self.token2id[action] = id + 1
        self.id2token = {v: k for k, v in self.token2id.items()}

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

    def add_hierarchy(self, general_token, id):
        if self.token2id[general_token] in self.action_hierarchy:
            self.action_hierarchy[self.token2id[general_token]].append(id)
        else:
            self.action_hierarchy[self.token2id[general_token]] = []
            self.action_hierarchy[self.token2id[general_token]].append(id)

    def __getitem__(self, token):
        general_token = type(token).__name__
        return self.token2id.get(general_token)


    @staticmethod
    def from_action_vocab(action_vocab_dict):
        vocab_entry = GeneralActionVocabEntry()
        for token, id in action_vocab_dict.items():
            general_token = type(token).__name__
            vocab_entry.add_token(general_token)
            vocab_entry.add_hierarchy(general_token, id)
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

