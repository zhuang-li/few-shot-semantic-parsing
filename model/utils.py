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
        new_tensor = embed_layer.weight.data.new
        word_ids = set(range(embed_layer.num_embeddings))
        no_update_ids = []
        for word, embed in self.embeddings:
            if word in vocab.token2id:
                word_id = vocab.token2id[word]
                word_ids.remove(word_id)
                embed_layer.weight[word_id].data.copy_(new_tensor(embed))
                no_update_ids.append(word_id)

        print ("non-initialized words")
        word_ids = list(word_ids)
        for word_id in word_ids:
            print (vocab.id2token[word_id])
        return no_update_ids

    def load_pre_train_to(self, embed_layer, token2id):
        new_tensor = embed_layer.weight.data.new
        word_ids = set(range(embed_layer.num_embeddings))
        count = 0
        for word, embed in self.embeddings:
            if word in token2id:
                word_id = token2id[word]
                word_ids.remove(word_id)
                embed_layer.weight[word_id].data.copy_(new_tensor(embed))
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