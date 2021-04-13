import collections
import itertools
import json
import os

import attr
import numpy as np
import torch

from model.rat_sql.models.spider import spider_enc_modules
from model.rat_sql.models.spider.spider_enc_modules import RelationalTransformerUpdate


@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]

class SpiderEncoderV2(torch.nn.Module):
    batched = True

    def __init__(
            self,
            device,
            vocab,
            embedding,
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.,
            question_encoder=('emb', 'bilstm'),
            column_encoder=('emb', 'bilstm-summarize'),
            include_in_memory=('question', 'column'),
            batch_encs_update=False):
        super().__init__()
        self._device = device
        self.embedding = embedding
        self.vocab = vocab
        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0
        #word_freq = self.preproc.vocab_builder.word_freq
        #top_k_words = set([_a[0] for _a in word_freq.most_common(top_k_learnable)])
        self.learnable_words = set(vocab.top_k_tokens)

        self.include_in_memory = set(include_in_memory)
        self.dropout = dropout
        #print (self.dropout)

        self.question_encoder = self._build_modules(question_encoder)
        self.column_encoder = self._build_modules(column_encoder)


        self.encs_update = RelationalTransformerUpdate(device, 4, 8, recurrent_size,
                 ff_size=None,
                 dropout=0.2,
                 merge_types=True,
                 tie_layers=False,
                 qq_max_dist=2,
                 cc_max_dist=2,
                 sc_link=True)

        self.batch_encs_update = batch_encs_update

    def _build_modules(self, module_types):
        module_builder = {
            'emb': lambda: spider_enc_modules.LookupEmbeddings(
                self._device,
                self.vocab,
                self.embedding,
                self.word_emb_size,
                self.learnable_words),
            'linear': lambda: spider_enc_modules.EmbLinear(
                input_size=self.word_emb_size,
                output_size=self.word_emb_size),
            'bilstm': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False),
            'bilstm-native': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=False,
                use_native=True),
            'bilstm-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True),
            'bilstm-native-summarize': lambda: spider_enc_modules.BiLSTM(
                input_size=self.word_emb_size,
                output_size=self.recurrent_size,
                dropout=self.dropout,
                summarize=True,
                use_native=True),
        }

        modules = []
        for module_type in module_types:
            modules.append(module_builder[module_type]())
        return torch.nn.Sequential(*modules)

    def forward_unbatched(self, desc):
        # Encode the question
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?

        # q_enc: question len x batch (=1) x recurrent_size
        q_enc, (_, _) = self.question_encoder([desc['question']])

        # Encode the columns
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each column into one?
        # c_enc: sum of column lens x batch (=1) x recurrent_size
        c_enc, c_boundaries = self.column_encoder(desc['column'])
        column_pointer_maps = {
            i: list(range(left, right))
            for i, (left, right) in enumerate(zip(c_boundaries, c_boundaries[1:]))
        }


        # Update each other using self-attention
        # q_enc_new, c_enc_new, and t_enc_new now have shape
        # batch (=1) x length x recurrent_size
        q_enc_new, c_enc_new, t_enc_new = self.encs_update(
            desc, q_enc, c_enc, c_boundaries)

        memory = []
        words_for_copying = []
        if 'question' in self.include_in_memory:
            memory.append(q_enc_new)
            if 'question_for_copying' in desc:
                assert q_enc_new.shape[1] == desc['question_for_copying']
                words_for_copying += desc['question_for_copying']
            else:
                words_for_copying += [''] * q_enc_new.shape[1]
        if 'column' in self.include_in_memory:
            memory.append(c_enc_new)
            words_for_copying += [''] * c_enc_new.shape[1]

        memory = torch.cat(memory, dim=1)

        return SpiderEncoderState(
            state=None,
            memory=memory,
            words=words_for_copying,
            pointer_memories={
                'column': c_enc_new,
            },
            pointer_maps={
                'column': column_pointer_maps
            }
        )

    def forward(self, descs):
        # Encode the question
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?

        # q_enc: PackedSequencePlus, [batch, question len, recurrent_size]
        qs = [[desc['question']] for desc in descs]
        #print (qs)
        q_enc, _ = self.question_encoder(qs)

        # Encode the columns
        # - LookupEmbeddings
        # - Transform embeddings wrt each other?
        # - Summarize each column into one?
        # c_enc: PackedSequencePlus, [batch, sum of column lens, recurrent_size]
        #print (type(self.column_encoder))
        #token_list = [desc['column'] for desc in descs]
        #print("token list")
        #print (token_list)
        c_enc, c_boundaries = self.column_encoder([desc['column'] for desc in descs])
        #print (c_enc.lengths)

        column_pointer_maps = [
            {
                i: list(range(left, right))
                for i, (left, right) in enumerate(zip(c_boundaries_for_item, c_boundaries_for_item[1:]))
            }
            for batch_idx, c_boundaries_for_item in enumerate(c_boundaries)
        ]



        # c_enc_lengths = list(c_enc.orig_lengths())
        # table_pointer_maps = [
        #     {
        #         i: [
        #             idx
        #             for col in desc['table_to_columns'][str(i)]
        #             for idx in column_pointer_maps[batch_idx][col]
        #         ] +  list(range(left + c_enc_lengths[batch_idx], right + c_enc_lengths[batch_idx]))
        #         for i, (left, right) in enumerate(zip(t_boundaries_for_item, t_boundaries_for_item[1:]))
        #     }
        #     for batch_idx, (desc, t_boundaries_for_item) in enumerate(zip(descs, t_boundaries))
        # ]

        # Update each other using self-attention
        # q_enc_new, c_enc_new, and t_enc_new are PackedSequencePlus with shape
        # batch (=1) x length x recurrent_size
        if self.batch_encs_update:
            q_enc_new, c_enc_new = self.encs_update(
                descs, q_enc, c_enc, c_boundaries)


        result = []
        for batch_idx, desc in enumerate(descs):
            #print (batch_idx)
            if self.batch_encs_update:
                q_enc_new_item = q_enc_new.select(batch_idx).unsqueeze(0)
                c_enc_new_item = c_enc_new.select(batch_idx).unsqueeze(0)
            else:
                q_enc_new_item, c_enc_new_item, align_mat_item = \
                    self.encs_update.forward_unbatched(
                        desc,
                        q_enc.select(batch_idx).unsqueeze(1),
                        c_enc.select(batch_idx).unsqueeze(1),
                        c_boundaries[batch_idx])

            #print (c_enc_new_item.size())
            memory = []
            words_for_copying = []
            if 'question' in self.include_in_memory:
                memory.append(q_enc_new_item)
                if 'question_for_copying' in desc:
                    assert q_enc_new_item.shape[1] == len(desc['question_for_copying'])
                    words_for_copying += desc['question_for_copying']
                else:
                    words_for_copying += [''] * q_enc_new_item.shape[1]
            if 'column' in self.include_in_memory:
                memory.append(c_enc_new_item)
                words_for_copying += [''] * c_enc_new_item.shape[1]

            memory = torch.cat(memory, dim=1)
            #print(q_enc_new_item.size())
            #print(c_enc_new_item.size())
            #print(memory.size())
            result.append(SpiderEncoderState(
                state=None,
                memory=memory,
                question_memory=q_enc_new_item,
                schema_memory=c_enc_new_item,
                # TODO: words should match memory
                words=words_for_copying,
                pointer_memories={
                    'column': c_enc_new_item
                },
                pointer_maps={
                    'column': column_pointer_maps[batch_idx]
                },
                m2c_align_mat=align_mat_item
            ))
        return result


