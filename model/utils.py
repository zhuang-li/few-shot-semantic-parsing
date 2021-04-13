# coding=utf-8


import math
import numpy as np


class GloveHelper(object):
    def __init__(self, glove_file, embed_size):
        self.glove_file = glove_file
        embeds = np.zeros((5000, embed_size), dtype='float32')
        for i, (word, embed) in enumerate(self.embeddings):
            if i == 5000: break
            embeds[i] = embed

        self.mean = np.mean(embeds)
        self.std = np.std(embeds)

    @property
    def embeddings(self):
        with open(self.glove_file, 'r') as f:
            for line in f:
                tokens = line.split()
                word, embed = tokens[0], np.array([float(tok) for tok in tokens[1:]])
                yield word, embed

    def emulate_embeddings(self, shape):
        samples = np.random.normal(self.mean, self.std, size=shape)

        return samples

    def load_to(self, embed_layer, vocab):
        #print (vocab.lemma_token2id)
        new_tensor = embed_layer.weight.data.new
        word_ids = set(range(embed_layer.num_embeddings))
        no_update_ids = []
        for word, embed in self.embeddings:
            if word in vocab.token2id:
                word_id = vocab.token2id[word]
                word_ids.remove(word_id)
                embed_layer.weight[word_id].data.copy_(new_tensor(embed))
                no_update_ids.append(word_id)
        word_ids = list(word_ids)
        for word_id in word_ids:
            print (vocab.id2token[word_id])
        return no_update_ids
        #embed_layer.weight[word_ids].data = new_tensor(self.emulate_embeddings(shape=(len(word_ids), embed_layer.embedding_dim)))

    def load_pre_train_to(self, embed_layer, token2id):
        new_tensor = embed_layer.weight.data.new
        word_ids = set(range(embed_layer.num_embeddings))
        count = 0
        for word, embed in self.embeddings:
            if word in token2id:
                word_id = token2id[word]
                word_ids.remove(word_id)
                embed_layer.weight[word_id].data.copy_(new_tensor(embed))
                print (word_id)
                count += 1
                if count == len(token2id):
                    break


    @property
    def words(self):
        with open(self.glove_file, 'r') as f:
            for line in f:
                tokens = line.split()
                yield tokens[0]


def batch_iter(examples, batch_size, shuffle=False):
    batch_num = int(math.ceil(len(examples) / float(batch_size)))
    index_array = list(range(len(examples)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_examples = [examples[idx] for idx in indices]

        yield batch_examples

def merge_vocab(pre_train_vocab_entry, new_vocab_entry,type):
    merged_token2id = dict()
    for token, id in pre_train_vocab_entry.token2id.items():
        if not token in merged_token2id:
            merged_token2id[token] = len(merged_token2id)

    for token, id in new_vocab_entry.token2id.items():
        if not token in merged_token2id:
            merged_token2id[token] = len(merged_token2id)

    merged_id2token = {v: k for k, v in merged_token2id.items()}
    new_vocab_entry.token2id = merged_token2id
    new_vocab_entry.id2token = merged_id2token

    if type == "t_action":

        merged_vertex2action = dict()
        for vertex, action in pre_train_vocab_entry.vertex2action.items():
            if not vertex in merged_vertex2action:
                merged_vertex2action[vertex] = action

        for vertex, action in new_vocab_entry.vertex2action.items():
            if not vertex in merged_vertex2action:
                merged_vertex2action[vertex] = action

        new_vocab_entry.vertex2action = merged_vertex2action

        merged_lhs2rhs = dict()
        for lhs, rhs in pre_train_vocab_entry.lhs2rhs.items():
            if not lhs in merged_lhs2rhs:
                merged_lhs2rhs[lhs] = rhs
            else:
                merged_lhs2rhs[lhs].extend(rhs)

        for lhs, rhs in new_vocab_entry.lhs2rhs.items():
            if not lhs in merged_lhs2rhs:
                merged_lhs2rhs[lhs] = rhs
            else:
                merged_lhs2rhs[lhs].extend(rhs)

        merged_rhs2lhs = dict()

        for lhs, rhs_list in merged_lhs2rhs.items():
            for rhs in rhs_list:
                merged_rhs2lhs[rhs] = lhs

        new_vocab_entry.lhs2rhs = merged_lhs2rhs
        new_vocab_entry.rhs2lhs = merged_rhs2lhs


        merged_lhs2rhsid = dict()
        for lhs, rhs_list in merged_lhs2rhs.items():
            rhsid = []
            for rhs in rhs_list:
                action = merged_vertex2action[rhs]
                id = merged_token2id[action]
                rhsid.append(id)
            merged_lhs2rhsid[lhs] = rhsid
        new_vocab_entry.lhs2rhsid = merged_lhs2rhsid


def merge_vocab_entry(pretrain_vocab, new_vocab):
    merge_vocab(pretrain_vocab.source, new_vocab.source, "source")
    merge_vocab(pretrain_vocab.nt_action, new_vocab.nt_action, "nt_action")
    merge_vocab(pretrain_vocab.t_action, new_vocab.t_action, "t_action")
    return new_vocab