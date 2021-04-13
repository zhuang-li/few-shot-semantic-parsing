# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch import _VF

from components.dataset import Batch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from common.registerable import Registrable
from model import nn_utils
from grammar.consts import *
from grammar.vertex import *
from grammar.rule import ReduceAction, GenAction, Rule
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
from model.locked_dropout import LockedDropout
import torch.nn.functional as F
from model.attention_util import AttentionUtil, Attention
import os

from model.rat_sql.models import attention
from model.rat_sql.models.spider.spider_dec_func import compute_align_loss
from model.rat_sql.models.spider.spider_enc import SpiderEncoderV2
from model.rat_sql.models.spider.spider_match_utils import compute_schema_linking


def copy_embedding_to_current_layer(pre_trained_vocab, current_vocab, pre_trained_layer, current_layer,
                                    pre_trained_mask=None, current_mask=None, train_action_set=None):
    if len(pre_trained_vocab) == len(current_vocab):
        print("vocab not need to update")
    else:
        print("update vocab pre-trained length {} and current length {}".format(len(pre_trained_vocab),
                                                                                len(current_vocab)))
    if train_action_set is not None:
        is_exist_vocab = train_action_set
    else:
        is_exist_vocab = pre_trained_vocab
    uninit_vocab = {}
    #print (is_exist_vocab)
    #print (pre_trained_vocab.token2id)
    for current_idx, current_token in current_vocab.id2token.items():
        if current_token in is_exist_vocab:
            #print (current_token)
            pre_trained_id = pre_trained_vocab[current_token]
            #print (pre_trained_id)
            #print (pre_trained_layer.weight.data[pre_trained_id].size())
            #print (pre_trained_layer.weight.data.size())
            #print (current_layer.weight.size())
            current_layer.weight.data[current_idx].copy_(pre_trained_layer.weight.data[pre_trained_id])
            if current_mask is not None:
                # print (current_mask.size())
                # print (pre_trained_mask.size())
                current_mask[0][current_idx] = pre_trained_mask[0][pre_trained_id]
        else:
            uninit_vocab[current_token] = current_idx
    #print(uninit_vocab)
    return uninit_vocab


def update_vocab_related_parameters(parser, loaded_vocab):
    parser.vocab = loaded_vocab

    pre_trained_src_vocab = parser.src_vocab
    parser.src_vocab = loaded_vocab.source

    pre_trained_src_embedding = parser.src_embed
    parser.src_embed = nn.Embedding(len(parser.src_vocab), parser.embed_size)
    nn.init.xavier_normal_(parser.src_embed.weight.data)

    print("update src embedding")
    unit_src_vocab = copy_embedding_to_current_layer(pre_trained_src_vocab, parser.src_vocab, pre_trained_src_embedding,
                                                     parser.src_embed)

    parser.encoder.embedding = parser.src_embed
    parser.encoder.question_encoder[0].embedding = parser.src_embed
    parser.encoder.column_encoder[0].embedding = parser.src_embed

    lemma_src_vocab = {}

    for token, token_id in unit_src_vocab.items():
        if token in STEM_LEXICON:
            lemma_src_vocab[STEM_LEXICON[token]] = token_id
        else:
            lemma_src_vocab[token] = token_id

    pre_trained_vertex_vocab = parser.vertex_vocab
    parser.vertex_vocab = loaded_vocab.vertex

    pre_trained_ast_embedding = parser.ast_embed
    parser.ast_embed = nn.Embedding(len(parser.vertex_vocab), parser.action_embed_size)
    nn.init.xavier_normal_(parser.ast_embed.weight.data)

    print("update ast embedding")
    unit_vertex_vocab = copy_embedding_to_current_layer(pre_trained_vertex_vocab, parser.vertex_vocab,
                                                        pre_trained_ast_embedding, parser.ast_embed)

    pre_trained_action_vocab = parser.action_vocab
    parser.action_vocab = loaded_vocab.action

    pre_trained_action_embedding = parser.readout_action
    parser.readout_action = nn.Linear(parser.hidden_size, len(parser.action_vocab), bias=False)
    nn.init.xavier_normal_(parser.readout_action.weight.data)


    print("update action embedding")
    pad_action = parser.vocab.action.id2token[0]
    pad_action.prototype_tokens = ['<pad>']

    start_action = parser.vocab.action.id2token[1]
    start_action.prototype_tokens = ['<s>']

    parser.predicate_tokens = [["and", str(action.rule.body_length)] if 'hidden' in action.prototype_tokens else action.prototype_tokens for action in parser.vocab.action.token2id.keys()]
    print(parser.predicate_tokens)

    copy_embedding_to_current_layer(pre_trained_action_vocab, parser.action_vocab, pre_trained_action_embedding,
                                    parser.readout_action)


    parser.action_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.action_vocab),
                                                                  ignore_indices=[0, 1])

    if len(parser.entity_vocab) == len(loaded_vocab.entity):
        print("entity vocab not need to update")
    else:
        print("===========================================")
        print(loaded_vocab.entity.token2id)
        print(parser.entity_vocab.token2id)
        pre_trained_entity_vocab = parser.entity_vocab
        parser.entity_vocab = loaded_vocab.entity
        pre_trained_readout_entity = parser.readout_entity
        parser.readout_entity = nn.Linear(parser.hidden_size, len(parser.entity_vocab), bias=False)
        nn.init.xavier_normal_(parser.readout_entity.weight.data)
        copy_embedding_to_current_layer(pre_trained_entity_vocab, parser.entity_vocab, pre_trained_readout_entity,
                                        parser.readout_entity)

        parser.entity_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.entity_vocab),
                                                                      ignore_indices=[0])
        print("update entity vocab pre-trained length {} and current length {}".format(len(loaded_vocab.entity),
                                                                                       len(parser.entity_vocab)))

    if len(parser.variable_vocab) == len(loaded_vocab.variable):
        print("variable vocab not need to update")
    else:
        print("update variable vocab pre-trained length {} and current length {}".format(len(loaded_vocab.variable),
                                                                                         len(parser.variable_vocab)))

    return lemma_src_vocab, unit_vertex_vocab

@Registrable.register('ratsql')
class Seq2SeqModel(nn.Module):
    """
    a standard seq2seq model
    """

    def __init__(self, vocab, args):
        super(Seq2SeqModel, self).__init__()
        self.use_cuda = args.use_cuda
        self.embed_size = args.embed_size
        self.action_embed_size = args.action_embed_size
        self.hidden_size = args.hidden_size
        self.vocab = vocab
        self.args = args
        self.src_vocab = vocab.source
        self.vertex_vocab = vocab.vertex
        self.action_vocab = vocab.action

        self.src_embed = nn.Embedding(len(self.src_vocab), self.embed_size)

        self.ast_embed = nn.Embedding(len(self.vertex_vocab), self.action_embed_size)

        nn.init.xavier_normal_(self.src_embed.weight.data)
        nn.init.xavier_normal_(self.ast_embed.weight.data)

        # general action embed
        self.entity_vocab = vocab.entity
        self.variable_vocab = vocab.variable

        # whether to use att
        self.use_att = args.use_att
        if args.use_att:
            self.decoder_size = self.action_embed_size + self.hidden_size
        else:
            self.decoder_size = self.action_embed_size

        self.encoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)

        self.decoder_lstm = nn.LSTMCell(self.decoder_size, self.hidden_size)

        self.use_input_lstm_encode = args.use_input_lstm_encode
        if self.use_input_lstm_encode:
            self.input_encode_lstm = nn.LSTM(self.decoder_size, self.decoder_size, bidirectional=False)


        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # supervised attention
        self.sup_attention = args.sup_attention

        # dropout layer
        self.dropout = args.dropout
        self.dropout_value = args.dropout
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout)

        self.dropout_i = args.dropout_i
        if self.dropout_i > 0:
            self.dropout_i = nn.Dropout(self.dropout_i)

        # project the input stack and reduce head into the embedding space
        self.input_stack_reduce_linear = nn.Linear(self.decoder_size * 2, self.decoder_size,
                                                   bias=False)

        self.readout_action = nn.Linear(self.hidden_size, len(self.action_vocab), bias=False)

        # entity prediction
        self.entity_lstm = nn.LSTMCell(self.decoder_size, self.hidden_size)

        # self.entity_embed = nn.Embedding(2, self.action_embed_size)
        # nn.init.xavier_normal_(self.entity_embed.weight.data)

        self.readout_entity = nn.Linear(self.hidden_size, len(self.entity_vocab), bias=False)

        # variable prediction
        self.variable_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.variable_embed = nn.Embedding(len(self.variable_vocab), self.hidden_size)

        nn.init.xavier_normal_(self.variable_embed.weight.data)

        self.readout_variable = nn.Linear(self.hidden_size, len(self.variable_vocab), bias=False)

        self.label_smoothing = None

        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            #print(len(self.action_vocab))
            self.action_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.action_vocab),
                                                                        ignore_indices=[0, 1])

            self.entity_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.entity_vocab),
                                                                        ignore_indices=[0])
            if len(self.variable_vocab) > 2:
                self.variable_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing,
                                                                              len(self.variable_vocab),
                                                                              ignore_indices=[0])

        pad_action = self.vocab.action.id2token[0]
        pad_action.prototype_tokens = ['<pad>']

        start_action = self.vocab.action.id2token[1]
        start_action.prototype_tokens = ['<s>']

        self.predicate_tokens = [None]*len(self.vocab.action.token2id.keys())

        for action, idx in self.vocab.action.token2id.items():
            if 'hidden' in action.prototype_tokens:
                self.predicate_tokens[idx] = ["and", str(action.rule.body_length)]
            else:
                self.predicate_tokens[idx] = action.prototype_tokens
        #print (self.predicate_tokens)
        #[["and", str(action.rule.body_length)] if 'hidden' in action.prototype_tokens else action.prototype_tokens for action in self.vocab.action.token2id.keys()]
        # [action.prototype_tokens for action in self.vocab.action.token2id.keys()]
        #print (self.predicate_tokens)

        self.encoder = SpiderEncoderV2(
                'cuda',
                vocab=self.src_vocab,
                embedding=self.src_embed,
                word_emb_size=self.embed_size,
                recurrent_size=self.hidden_size,
                dropout=self.dropout_value,
                batch_encs_update=False)

        self.pointer_attention = attention.ScaledDotProductPointer(
                query_size=self.hidden_size,
                key_size=self.hidden_size)

        self.question_attn = attention.MultiHeadedAttention(
            h=1,
            query_size=self.hidden_size,
            value_size=self.hidden_size)
        self.schema_attn = attention.MultiHeadedAttention(
            h=1,
            query_size=self.hidden_size,
            value_size=self.hidden_size)

        self.use_coverage = args.use_coverage
        if self.use_coverage:
            self.cov_linear = nn.Linear(1, self.hidden_size, bias=False)
            self.fertility = nn.Linear(self.hidden_size, 1, bias=False)

        if args.use_cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor


    def decode(self, desc, dec_init_vec, batch):
        """
        compute the final softmax layer at each decoding step
        :param src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
        :param src_sents_len: list[int]
        :param dec_init_vec: tuple((batch_size, hidden_size))
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            scores: Variable(src_sent_len, batch_size, src_vocab_size)
        """

        action_seq = batch.action_seq
        bos = GenAction(RuleVertex('<s>'))
        action_seq = [[bos] + seq for seq in action_seq]

        ast_seq = batch.ast_seq
        ast_seq_var = batch.ast_seq_var

        assert len(ast_seq[0]) + 1 == len(
            action_seq[0]), "length of sequence variable should be the same with the length of the action sequence"

        #new_tensor = src_encodings.data.new
        #batch_size = src_encodings.size(1)

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size * 2)
        #src_encodings = src_encodings.permute(1, 0, 2)
        # print (src_encodings.size())
        # (batch_size, query_len, hidden_size)
        #src_encodings = desc.question_memory
        #src_encodings_att_linear = self.att_src_linear(src_encodings)

        # initialize the attentional vector
        att_tm1 = self.new_tensor(1, self.hidden_size).zero_()
        assert att_tm1.requires_grad == False, "the att_tm1 requires grad is False"
        # (batch_size, src_sent_len)

        # print (src_sent_masks.size())
        # (tgt_sent_len, batch_size, embed_size)

        ast_seq_embed = self.ast_embed(ast_seq_var)
        if len(batch.variable_seq[0]) > 0:
            var_seq_embed = self.variable_embed(batch.variable_seq_var)


        # if len(batch.entity_seq[0]) > 0:
        # entity_seq_embed = self.entity_embed(batch.entity_seq_var)

        scores = []
        variable_scores = []
        entity_scores = []
        att_probs = []
        hyp = Hypothesis()
        hyp.hidden_embedding_stack.append(h_tm1)
        hyp.v_hidden_embedding = h_tm1


        v_id = 0
        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(ast_seq_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim



            y_tm1_embed = y_tm1_embed.squeeze(0)

            # action instance
            current_node = action_seq[0][t]

            # if self.dropout_i:
            # y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)

            if self.use_att:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed



            if len(current_node.entities) > 0:
                h_tm1_e = h_tm1
                # e_id = 0
                for ent in current_node.entities:
                    # print (x.size())
                    # h_tm1_e = self.entity_lstm(x, h_tm1_e)

                    (h_e, cell_e), att_t_e, att_weight_e = self.step(desc, x, h_tm1_e, lstm_type='entity')

                    score_e_t = self.readout_entity(h_e)


                    score_e = torch.log_softmax(score_e_t, dim=-1)
                    entity_scores.append(score_e)
                    h_tm1_e = (h_e, cell_e)
                    # e_id += 1

            if len(current_node.variables) > 0:
                #h_tm1_v = h_tm1
                input_v = att_tm1
                for var in current_node.variables:
                    # print (x.size())
                    #hyp.v_hidden_embedding = self.variable_lstm(input_v, hyp.v_hidden_embedding)
                    (h_t, cell_t), att_t, att_weight = self.step(desc, input_v, hyp.v_hidden_embedding,
                                                                 lstm_type='variable')

                    score_v_t = self.readout_variable(h_t)


                    score_v = torch.log_softmax(score_v_t, dim=-1)
                    variable_scores.append(score_v)
                    input_v = var_seq_embed[v_id]
                    v_id += 1
                    hyp.v_hidden_embedding = (h_t, cell_t)
            # if self.dropout_i:
            # x = self.dropout_i(x.unsqueeze(0)).squeeze(0)

            (h_t, cell_t), att_t, att_weight = self.step(desc, x, h_tm1)
            # (batch_size, tgt_vocab_size)
            #print(desc.words)
            #print(att_t.size())
            #print(desc.memory.size())
            memory_pointer_logits = self.pointer_attention(
                h_t, desc.memory)
            #print(memory_pointer_logits.size())
            memory_pointer_probs = torch.nn.functional.softmax(
                memory_pointer_logits, dim=1)
            #print(memory_pointer_probs.size())
            # pointer_logits shape: batch (=1) x num choices
            pointer_probs = torch.mm(memory_pointer_probs, desc.m2c_align_mat)
            #print(pointer_probs.size())
            #print(pointer_probs.sum())
            pointer_probs = pointer_probs.clamp(min=1e-9)
            score = torch.log(pointer_probs)

            # self.readout_action(att_t)  # E.q. (6)

            # print (re_score_t)
            #score = torch.log_softmax(score_t, dim=-1)
            scores.append(score)
            # cov_v = cov_v + att_weight
            if self.sup_attention:
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_actions):
                        action_t = example.tgt_actions[t]

                        # assert str(action_t) == str(
                        # current_node), "for consistence of the current input and the output actions"
                        cand_src_tokens = AttentionUtil.get_candidate_tokens_to_attend(example.src_sent, action_t)
                        # print (example.src_sent)
                        # print (action_t)
                        # print (cand_src_tokens)
                        if cand_src_tokens:
                            # print (att_weight[e_id].size())
                            # print (example.src_sent)
                            # print (att_weight[e_id])
                            # print (cand_src_tokens)
                            att_prob = [torch.log(att_weight[e_id, token_id + 1].unsqueeze(0)) for token_id in
                                        cand_src_tokens]
                            # print (cand_src_tokens)
                            # print (att_prob)
                            if len(att_prob) > 1:
                                att_prob = torch.cat(att_prob).sum().unsqueeze(0)
                            else:
                                att_prob = att_prob[0]

                            att_probs.append(att_prob)
            hyp.hidden_embedding_stack.append((h_t, cell_t))

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        scores = torch.stack(scores)
        if len(entity_scores) > 0:
            entity_scores = torch.stack(entity_scores)
        else:
            entity_scores = None
        if len(variable_scores) > 0:
            variable_scores = torch.stack(variable_scores)
        else:
            variable_scores = None

        if len(att_probs) > 0:
            att_probs = torch.stack(att_probs).sum()
        else:
            att_probs = None
        # print (att_probs.size())
        if self.sup_attention:
            return (scores, entity_scores, variable_scores), att_probs
        else:
            return (scores, entity_scores, variable_scores)

    def score_decoding_results(self, scores, batch):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        # (tgt_sent_len, batch_size, tgt_vocab_size)

        action_scores, entity_scores, variable_scores = scores
        action_var = batch.action_seq_var[1:]
        if entity_scores is not None:
            entity_var = batch.entity_seq_var
        if variable_scores is not None:
            variable_var = batch.variable_seq_var
        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            # print (nn_scores.size())
            # print (gen_action_var.size())
            # print (re_scores.size())
            # print (re_action_var.size())
            #print (action_scores.size())
            #print (action_var.size())
            #print (action_scores.size())
            #print (action_var)
            sent_log_scores = -self.action_label_smoothing_layer(action_scores, action_var)
            if entity_scores is not None:
                entity_log_scores = -self.entity_label_smoothing_layer(entity_scores, entity_var)
            if variable_scores is not None:
                if len(self.variable_vocab) > 2:
                    variable_log_scores = -self.variable_label_smoothing_layer(variable_scores, variable_var)
                else:
                    variable_log_scores = torch.gather(variable_scores, -1, variable_var.unsqueeze(-1)).squeeze(-1)

        else:
            # (tgt_sent_len, batch_size)
            sent_log_scores = torch.gather(action_scores, -1, action_var.unsqueeze(-1)).squeeze(-1)
            if entity_scores is not None:
                entity_log_scores = torch.gather(entity_scores, -1, entity_var.unsqueeze(-1)).squeeze(-1)
            if variable_scores is not None:
                variable_log_scores = torch.gather(variable_scores, -1, variable_var.unsqueeze(-1)).squeeze(-1)

        sent_log_scores = sent_log_scores * (1. - torch.eq(action_var, 0).float())  # 0 is pad

        sent_log_scores = sent_log_scores.sum(dim=0)
        # print (gen_sent_log_scores)

        # generate action loss
        loss = -sent_log_scores.unsqueeze(0)[0]
        if entity_scores is not None:
            entity_log_scores = entity_log_scores * (1. - torch.eq(entity_var, 0).float())  # 0 is pad

            entity_log_scores = entity_log_scores.sum(dim=0)

            entity_loss = -entity_log_scores.unsqueeze(0)[0]
        if variable_scores is not None:
            variable_log_scores = variable_log_scores * (1. - torch.eq(variable_var, 0).float())  # 0 is pad

            variable_log_scores = variable_log_scores.sum(dim=0)

            variable_loss = -variable_log_scores.unsqueeze(0)[0]

        if entity_scores is not None:
            loss = loss + entity_loss
        if variable_scores is not None:
            loss = loss + variable_loss
        return loss

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        dec_init_cell = enc_last_cell
        #print (dec_init_cell.squeeze().size())
        dec_init_state = torch.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]

        question_context, question_attention_logits = self.question_attn(query, desc_enc.question_memory)

        schema_context, schema_attention_logits = self.schema_attn(query, desc_enc.schema_memory)
        return question_context + schema_context, question_attention_logits.squeeze().unsqueeze(0)

    def step(self, desc, x, h_tm1, lstm_type='decoder'):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)

        if lstm_type == 'decoder':
            h_t, cell_t = self.decoder_lstm(x, h_tm1)
        elif lstm_type == 'entity':
            h_t, cell_t = self.entity_lstm(x, h_tm1)
        elif lstm_type == 'variable':
            h_t, cell_t = self.variable_lstm(x, h_tm1)

        ctx_t, alpha_t = self._desc_attention(h_tm1, desc)
        #print (torch.cat([h_t, ctx_t], 1).size())
        #print (alpha_t.size())
        #print (self.att_vec_linear(torch.cat([h_t, ctx_t], 1)).size())
        # vector for action prediction
        #att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        #if self.dropout:
            #att_t = self.dropout(att_t)

        return (h_t, cell_t), ctx_t, alpha_t

    def forward(self, examples):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """
        input = []

        #src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda, data_type='lambda')
        #print (examples)
        #print (predicate_tokens)
        for e in examples:
            desc = {'question': None, 'question_for_copying': None, 'sc_link': None, 'columns': None}
            seq = ['<s>'] + e.src_sent + ['</s>']
            desc['sc_link'] = compute_schema_linking(seq, self.predicate_tokens)
            desc['question'] = seq
            desc['question_for_copying'] = seq
            desc['column'] = self.predicate_tokens
            #print (desc)
            input.append(desc)

        state_results = self.encoder(input)
        #print (state_results[0].words)
        #print (state_results[0].question_memory.size())
        #print (state_results[0].schema_memory.size())



        #src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda, data_type='lambda')
        #src_sents_var = src_batch.src_sents_var
        #src_sents_len = src_batch.src_sents_len
        # action_seq_var = batch.action_seq_var

        #src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)

        #c0 = self.new_tensor(1, self.hidden_size).zero_()
        #self.init_decoder_state(last_state, last_cell)
        #print (dec_init_vec[0].size())



        #src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)
        loss = []
        loss_sup = []
        # print (len(src_batch))
        for i in range(len(examples)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda, data_type='lambda')
            #print("=====================================")
            #print(state_results[i].words)
            #print (state_results[i].question_memory.size())
            h0, c0 = self.init_decoder_state(state_results[i].question_memory[:,-1,:], state_results[i].question_memory[:,-1,:])

            if self.sup_attention:
                (action_scores, entity_scores, variable_scores), att_prob = self.decode(state_results[i], (h0, c0), tgt_batch)
                if att_prob:
                    loss_sup.append(att_prob)
            else:
                (action_scores, entity_scores, variable_scores) = self.decode(state_results[i], (h0, c0),
                                                                              tgt_batch)
            loss_s = self.score_decoding_results((action_scores, entity_scores, variable_scores), tgt_batch)

            #loss_s += compute_align_loss(state_results[i])

            loss.append(loss_s)
        ret_val = [torch.stack(loss)]
        if self.sup_attention:
            ret_val.append(loss_sup)

        # print(loss.data)
        return ret_val

    def assign_nodes_back(self, node, padding, origin_node_list, c = 0):

        if node.has_children():
            for child in node.children:
            # print (hyp.heads_stack[-1])
            # print (child)
                c = self.assign_nodes_back(child, padding, origin_node_list, c)
        else:
            if isinstance(node, RuleVertex):
                if node.head.startswith(padding):
                    node.head = origin_node_list[c]
                    c += 1
            else:
                if node.vertex.head.startswith(padding):
                    node.head = origin_node_list[c]
                    c += 1
        return c

    def beam_search(self, example, decode_max_time_step, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        # print ("============================")
        input = []

        #src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda, data_type='lambda')
        #print (examples)
        #print (predicate_tokens)
        e = example
        input_dict = {'question': None, 'question_for_copying': None, 'sc_link': None, 'columns': None}
        input_dict['sc_link'] = compute_schema_linking(e.src_sent, self.predicate_tokens)
        input_dict['question'] = e.src_sent
        input_dict['question_for_copying'] = e.src_sent
        input_dict['column'] = self.predicate_tokens
        #print (desc)
        input.append(input_dict)

        state_results = self.encoder(input)

        desc = state_results[0]

        h_tm1 = self.init_decoder_state(desc.question_memory[:,-1,:], desc.question_memory[:,-1,:])

        if self.use_cuda:
            new_long_tensor = torch.cuda.LongTensor
        else:
            new_long_tensor = torch.LongTensor

        att_tm1 = torch.zeros(1, self.hidden_size, requires_grad=True)
        hyp_scores = torch.zeros(1, requires_grad=True)

        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        # todo change it back
        # eos_id = self.action_vocab['</s>']
        node = RuleVertex('<s>')
        bos_id = self.vertex_vocab[node]
        vocab_size = len(self.action_vocab)

        first_hyp = Hypothesis()
        first_hyp.tgt_ids_stack.append(bos_id)
        # first_hyp.embedding_stack.append(att_tm1)
        first_hyp.hidden_embedding_stack.append(h_tm1)

        first_hyp.var_id.append(1)
        first_hyp.ent_id.append(1)
        first_hyp.v_hidden_embedding = h_tm1
            # hypotheses = [[bos_id]]
        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step and len(hypotheses) > 0:
            # if t == 50:
            # print (t)
            # hyp_num = len(hypotheses)


            y_tm1 = new_long_tensor([hyp.tgt_ids_stack[-1] for hyp in hypotheses])
            y_tm1_embed = self.ast_embed(y_tm1)
            """
            if self.dropout_i:
                y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)
            """
            if self.use_att:
                # print (hyp_num)
                # print (y_tm1_embed.size())
                # print (att_tm1.size())
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            # assert not torch.isnan(x).any(), "x has no nan"
            if not t == 0:
                for id, hyp in enumerate(hypotheses):
                    if len(hyp.actions[-1].entities) > 0:
                        h_tm1_e = (h_tm1[0][id].unsqueeze(0), h_tm1[1][id].unsqueeze(0))
                        ent_list = []
                        # argument_id = 0
                        for ent in hyp.actions[-1].entities:
                            # print (len(x_list))
                            # print (x_list[id].size())
                            # print (h_tm1_v[0].size())
                            # h_t_e, cell_t_e = self.entity_lstm(x[id].unsqueeze(0), h_tm1_e)

                            (h_t_e, cell_t_e), att_t_e, att_weight_e = self.step(desc, x[id].unsqueeze(0), h_tm1_e, lstm_type='entity')


                            score_e_t = self.readout_entity(h_t_e)
                            score_e = torch.log_softmax(score_e_t, dim=-1)
                            # print (score_v)
                            score_e[0][0] = -1000.0
                            #score_e[0][1] = -1000.0
                            _, _ent_prev_word = score_e.max(1)
                            # print (_var_prev_word)
                            e_id = _ent_prev_word.item()
                            hyp.ent_id.append(e_id)
                            ent_list.append(self.entity_vocab.id2token[e_id])
                            # argument_id += 1
                            h_tm1_e = (h_t_e, cell_t_e)

                        ent_node = hyp.heads_stack[-1] if isinstance(hyp.heads_stack[-1], RuleVertex) else \
                        hyp.heads_stack[
                            -1].vertex
                        if isinstance(hyp.actions[-1], ReduceAction):

                            c = 0
                            for child in ent_node.children:
                                # print (hyp.heads_stack[-1])
                                # print (child)
                                if isinstance(child, RuleVertex):
                                    if child.head.startswith(SLOT_PREFIX):
                                        child.head = ent_list[c]
                                        c += 1
                                else:
                                    if child.vertex.head.startswith(SLOT_PREFIX):
                                        child.head = ent_list[c]
                                        c += 1
                        else:
                            c = self.assign_nodes_back(ent_node, SLOT_PREFIX, ent_list, c=0)

                            # print (child)
                        assert c == len(hyp.actions[
                                            -1].entities), "variables {} and nodes variables {} num must be in consistency {} and {}".format(
                            c, len(hyp.actions[-1].entities), hyp.heads_stack[-1], hyp.actions[-1])

                    if len(hyp.actions[-1].variables) > 0:
                        #h_tm1_v = (h_tm1[0][id].unsqueeze(0), h_tm1[1][id].unsqueeze(0))
                        var_list = []
                        input_v = att_tm1[id].unsqueeze(0)
                        for var in hyp.actions[-1].variables:
                            # print (len(x_list))
                            # print (x_list[id].size())
                            # print (h_tm1_v[0].size())
                            #h_t, cell_t = self.variable_lstm(input_v,
                             #                                hyp.v_hidden_embedding)
                            (h_t, cell_t), att_t, att_weight = self.step(desc, input_v, hyp.v_hidden_embedding, lstm_type='variable')


                            score_v_t = self.readout_variable(h_t)
                            score_v = torch.log_softmax(score_v_t, dim=-1)
                            # print (score_v)
                            score_v[0][0] = -1000.0
                            #score_v[0][1] = -1000.0
                            _, _var_prev_word = score_v.max(1)
                            # print (_var_prev_word)
                            v_id = _var_prev_word.item()
                            hyp.var_id.append(v_id)
                            var_list.append(self.variable_vocab.id2token[v_id])
                            hyp.v_hidden_embedding = (h_t, cell_t)
                            input_v = self.variable_embed(new_long_tensor([hyp.var_id[-1]]))

                        var_node = hyp.heads_stack[-1] if isinstance(hyp.heads_stack[-1], RuleVertex) else \
                        hyp.heads_stack[
                            -1].vertex


                        if isinstance(hyp.actions[-1], ReduceAction):
                            c = 0
                            for child in var_node.children:
                                # print (hyp.heads_stack[-1])
                                # print (child)
                                if isinstance(child, RuleVertex):
                                    if child.head == VAR_NAME:
                                        child.head = var_list[c]
                                        c += 1
                                else:
                                    if child.vertex.head == VAR_NAME:
                                        child.head = var_list[c]
                                        c += 1
                        else:
                            c = self.assign_nodes_back(var_node, VAR_NAME, var_list, c=0)

                        assert c == len(hyp.actions[
                                            -1].variables), "variables {} and nodes variables {} num must be in consistency {} and {}".format(
                            c, len(hyp.actions[-1].variables), hyp.heads_stack[-1], hyp.actions[-1])


            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, att_weight = self.step(desc, x, h_tm1)


            memory_pointer_logits = self.pointer_attention(
                h_t, desc.memory)
            #print(memory_pointer_logits.size())
            memory_pointer_probs = torch.nn.functional.softmax(
                memory_pointer_logits, dim=1)
            #print(memory_pointer_probs.size())
            # pointer_logits shape: batch (=1) x num choices
            pointer_probs = torch.mm(memory_pointer_probs, desc.m2c_align_mat)
            #print(pointer_probs.size())
            #print(pointer_probs.sum())
            pointer_probs = pointer_probs.clamp(min=1e-9)
            p_t = torch.log(pointer_probs)

            #score_t = self.readout_action(att_t)  # E.q. (6)

            # hyp.current_gen_emb = current_att_t

            #p_t = torch.log_softmax(score_t, dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            # print(new_hyp_scores)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos // vocab_size
            word_ids = top_new_hyp_pos % vocab_size
            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data.tolist(),
                                                           word_ids.cpu().data.tolist(),
                                                           top_new_hyp_scores.cpu().data.tolist()):
                temp_hyp = hypotheses[prev_hyp_id].copy()

                action_instance = self.action_vocab.id2token[word_id]
                # print (action_instance)
                if isinstance(action_instance, ReduceAction):
                    temp_hyp.reduce_action_count = temp_hyp.reduce_action_count + 1
                    predict_body_length = action_instance.rule.body_length
                    # print (self.re_vocab.body_len_list.index(predict_body_length))
                    # print (self.re_vocab.body_len_list)
                    # print(predict_body_length)word_id == self.action_vocab[
                    #                         ReduceAction(Rule(RuleVertex('<re_pad>')))] or
                    if t == 0 or temp_hyp.reduce_action_count > 10:
                        # print ("continue")
                        continue

                    # temp_hyp.current_att = temp_hyp.current_re_emb[
                    # self.re_vocab.body_len_list.index(predict_body_length)].view(temp_hyp.current_gen_emb.size())

                    # print (temp_hyp.current_att.size())
                    if predict_body_length == FULL_STACK_LENGTH:
                        predict_body_length = len(temp_hyp.heads_stack)

                    if predict_body_length > len(temp_hyp.heads_stack):
                        continue
                    # print (prev_hyp_id)
                    # print (len(temp_hyp.embedding_stack))
                    temp_hyp.actions.append(action_instance)
                    temp_hyp.heads_stack = temp_hyp.reduce_actions(action_instance)
                    if temp_hyp.completed():

                        temp_hyp.tgt_ids_stack.append(self.vertex_vocab[action_instance.rule.head.copy()])
                        # temp_hyp.actions.append(re_action_instance)

                        # temp_hyp.embedding_stack = temp_hyp.embedding_stack[:-predict_body_length]
                        # temp_hyp.embedding_stack.append(temp_hyp.current_att)
                        # print (len(temp_hyp.hidden_embedding_stack))
                        temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))

                        temp_hyp.tgt_ids_stack = temp_hyp.tgt_ids_stack[1:]
                        temp_hyp.actions = temp_hyp.actions
                        temp_hyp.score = new_hyp_score
                        # new_hypotheses.append(temp_hyp)
                        completed_hypotheses.append(temp_hyp)
                    else:
                        temp_hyp.tgt_ids_stack.append(self.vertex_vocab[action_instance.rule.head.copy()])
                        # temp_hyp.actions.append(re_action_instance)

                        # temp_hyp.embedding_stack = temp_hyp.embedding_stack[:-predict_body_length]
                        # temp_hyp.embedding_stack.append(temp_hyp.current_att)
                        temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))
                        new_hypotheses.append(temp_hyp)
                        live_hyp_ids.append(prev_hyp_id)
                        new_hyp_scores.append(new_hyp_score)
                else:
                    temp_hyp.reduce_action_count = 0
                    if word_id == self.action_vocab[GenAction(RuleVertex('<gen_pad>'))] or word_id == self.action_vocab[
                        GenAction(RuleVertex('<s>'))]:
                        # print ("continue")
                        continue

                    temp_hyp.actions.append(action_instance)
                    temp_hyp.heads_stack.append(action_instance.vertex.copy())
                    temp_hyp.tgt_ids_stack.append(self.vertex_vocab[action_instance.vertex.copy()])
                    temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))
                    # print(len(hyp.hidden_embedding_stack))
                    # temp_hyp.current_att = temp_hyp.current_gen_emb
                    # temp_hyp.embedding_stack.append(temp_hyp.current_att)
                    new_hypotheses.append(temp_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)
            # print(new_hypotheses[0].actions)
            # print(new_hypotheses[0].actions[-1])
            if len(completed_hypotheses) == beam_size or len(new_hypotheses) == 0:
                break

            # live_hyp_ids = new_long_tensor(live_hyp_ids)
            # for emb in att_tm1_list:
            # print (emb.size())

            # att_tm1 = torch.stack(att_tm1_list)
            # print (att_tm1.size())
            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]
            att_w_tm1 = att_weight[live_hyp_ids]
            hyp_scores = self.new_tensor(new_hyp_scores)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
            t += 1
        if len(completed_hypotheses) == 0:
            """
            print ("======================= no parsed result !!! =================================")
            print(" ".join(example.src_sent))
            print (example.tgt_code_no_var_str)
            print("======================= no parsed result !!! =================================")
            """
            dummy_hyp = Hypothesis()
            completed_hypotheses.append(dummy_hyp)
        else:
            # completed_hypotheses = [hyp for hyp in completed_hypotheses if hyp.completed()]
            # todo: check the rank order
            completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # print ("============================")
        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        params = {
            'state_dict': self.state_dict(),
            'args': self.args,
            'vocab': self.vocab
        }

        torch.save(params, path)

    @classmethod
    def load(cls, model_path, use_cuda=False, loaded_vocab = None, args = None):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']

        saved_args.use_cuda = use_cuda

        saved_args.dropout = args.dropout
        saved_args.lr = args.lr
        saved_args.label_smoothing = args.label_smoothing
        saved_args.sup_attention = args.sup_attention

        parser = cls(vocab, saved_args)

        parser.load_state_dict(saved_state)
        src_vocab = vertex_vocab = None
        if loaded_vocab:
            src_vocab, vertex_vocab = update_vocab_related_parameters(parser, loaded_vocab)
        if use_cuda: parser = parser.cuda()
        parser.eval()

        return parser, src_vocab, vertex_vocab
