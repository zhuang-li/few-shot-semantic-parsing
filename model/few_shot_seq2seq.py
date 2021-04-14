# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch.distributions import Categorical

from components.dataset import Batch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from common.registerable import Registrable
from grammar.utils import is_lit
from model import nn_utils
from grammar.consts import *
from grammar.vertex import *
from grammar.rule import ReduceAction, GenAction, Rule
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
from model.attention_util import AttentionUtil
import os
import math
from model.nn_utils import masked_log_softmax


def copy_embedding_to_current_layer(pre_trained_vocab, current_vocab, pre_trained_layer, current_layer,
                                    pre_trained_mask=None, current_mask=None):
    if len(pre_trained_vocab) == len(current_vocab):
        print("vocab not need to update")
    else:
        print("update vocab pre-trained length {} and current length {}".format(len(pre_trained_vocab),
                                                                           len(current_vocab)))

    uninit_vocab = {}

    for current_idx, current_token in current_vocab.id2token.items():
        if current_token in pre_trained_vocab:
            pre_trained_id = pre_trained_vocab[current_token]
            current_layer.weight.data[current_idx].copy_(pre_trained_layer.weight.data[pre_trained_id])
            if current_mask is not None:

                current_mask[0][current_idx] = pre_trained_mask[0][pre_trained_id]
        else:
            uninit_vocab[current_token] = current_idx

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
    pre_trained_vertex_vocab = parser.vertex_vocab
    parser.vertex_vocab = loaded_vocab.vertex

    pre_trained_ast_embedding = parser.ast_embed
    parser.ast_embed = nn.Embedding(len(parser.vertex_vocab), parser.action_embed_size)
    nn.init.xavier_normal_(parser.ast_embed.weight.data)
    print("update ast embedding")
    copy_embedding_to_current_layer(pre_trained_vertex_vocab, parser.vertex_vocab,
                                                        pre_trained_ast_embedding, parser.ast_embed)

    parser.train_action_set.add(GenAction(RuleVertex('<gen_pad>')))
    parser.train_action_set.add(GenAction(RuleVertex('<s>')))
    pre_trained_action_vocab = parser.action_vocab
    print ("pretrained vocab", pre_trained_action_vocab)
    print ("new vocab", loaded_vocab.action)
    parser.action_vocab = loaded_vocab.action

    pre_trained_action_embedding = parser.action_embedding
    parser.action_embedding = nn.Embedding(len(parser.action_vocab), parser.hidden_size)
    nn.init.xavier_normal_(parser.action_embedding.weight.data)
    parser.action_proto = parser.new_tensor(len(parser.action_vocab), parser.hidden_size)
    pre_trained_action_mask = parser.action_mask
    parser.action_mask = torch.zeros(1, len(parser.action_vocab))

    print("update action embedding")
    copy_embedding_to_current_layer(pre_trained_action_vocab, parser.action_vocab, pre_trained_action_embedding,
                                parser.action_embedding, pre_trained_action_mask, parser.action_mask)

    print("action mask")
    print(parser.action_mask)

    parser.action_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.action_vocab),
                                                                  ignore_indices=[0, 1])


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

    if parser.use_cuda:
        parser.action_mask = parser.action_mask.cuda()
    return unit_src_vocab


@Registrable.register('seq2seq_few_shot')
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
        self.few_shot_mode = 'supervised_train'
        self.metric = args.metric
        self.hyp_embed = args.hyp_embed

        action_vocab_size = len(self.action_vocab)

        vertex_vocab_size = len(self.vertex_vocab)

        print ("action vocab size",action_vocab_size)
        print ("vertex vocab size",vertex_vocab_size)

        self.action_embedding = nn.Embedding(action_vocab_size, self.hidden_size)

        self.ast_embed = nn.Embedding(vertex_vocab_size, self.action_embed_size)

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
        self.att_vec_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size, bias=False)

        # supervised attention
        self.sup_attention = args.sup_attention

        # dropout layer
        self.dropout = args.dropout
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout)

        self.att_reg = args.att_reg
        if self.att_reg:
            self.att_reg_linear = nn.Linear(self.hidden_size, 1)
        # project the input stack and reduce head into the embedding space
        self.input_stack_reduce_linear = nn.Linear(self.decoder_size * 2, self.decoder_size,
                                                   bias=False)

        # entity prediction
        self.entity_lstm = nn.LSTMCell(self.decoder_size, self.hidden_size)

        self.readout_entity = nn.Linear(self.hidden_size, len(self.entity_vocab), bias=False)
        # variable prediction
        self.variable_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.variable_embed = nn.Embedding(len(self.variable_vocab), self.hidden_size)
        nn.init.xavier_normal_(self.variable_embed.weight.data)

        self.readout_variable = nn.Linear(self.hidden_size, len(self.variable_vocab), bias=False)
        self.action_freq = dict()
        self.hinge_loss = args.hinge_loss
        if self.hinge_loss:
            self.hinge_loss = torch.nn.MultiMarginLoss()

        self.label_smoothing = None

        self.entity_linear = nn.Linear(self.hidden_size, 1)
        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.action_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.action_vocab),
                                                                        ignore_indices=[0, 1])

            self.entity_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.entity_vocab),
                                                                        ignore_indices=[0])
            if len(self.variable_vocab) > 2:
                self.variable_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing,
                                                                              len(self.variable_vocab),
                                                                              ignore_indices=[0])

        if args.use_cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor


        nn.init.xavier_normal_(self.action_embedding.weight.data)
        self.action_proto = self.new_tensor(len(self.action_vocab), self.hidden_size)
        self.train_action_set = set()

        self.action_mask = torch.ones(1, len(self.action_vocab))
        self.train_mask = args.train_mask
        self.proto_mask = None
        if args.use_cuda:
            self.action_mask = self.action_mask.cuda()


    def init_action_embedding(self, examples, few_shot_mode='proto_train'):
        src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        src_sents_var = src_batch.src_sents_var
        src_sents_len = src_batch.src_sents_len
        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)

        att_logits_list = []
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda)
            res_dict = self.decode(
                src_encodings[:, i, :].unsqueeze(1),
                src_sent_masks[i].unsqueeze(0), (
                    dec_init_vec[0][i, :].unsqueeze(0),
                    dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
            att_logits_list.append(res_dict['att_logits'])

        new_action_proto = self.new_tensor(self.action_proto.size()).zero_()
        action_count = {}
        neo_type = 0
        action_set = set()

        for template_id, template_seq in enumerate(src_batch.action_seq):
            att_seq = att_logits_list[template_id]
            for att_id, action in enumerate(template_seq):
                action_id = self.action_vocab[action]

                if few_shot_mode == 'proto_train':
                    action_set.add(action)
                    new_action_proto[action_id] = new_action_proto[action_id] + att_seq[att_id]

                    if action_id in action_count:
                        action_count[action_id] += 1
                    else:
                        action_count[action_id] = 1
                elif few_shot_mode == 'fine_tune':
                    if not (action in self.train_action_set):
                        neo_type += 1
                        new_action_proto[action_id] = new_action_proto[action_id] + att_seq[att_id]

                        if action_id in action_count:
                            action_count[action_id] += 1
                        else:
                            action_count[action_id] = 1

        for action_vocab_id, freq in action_count.items():
            new_action_proto[action_vocab_id] = new_action_proto[action_vocab_id] / freq

            assert (new_action_proto[action_vocab_id].requires_grad == True)

        if few_shot_mode == 'proto_train':
            self.action_proto = new_action_proto
        elif few_shot_mode == 'fine_tune':
            self.action_embedding.weight.data.copy_(
                new_action_proto + (self.action_embedding.weight.data.t() * self.action_mask).t())
        return action_set

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask is not None:
            att_weight.data.masked_fill_(mask, -float('inf'))

        att_weight = torch.softmax(att_weight, dim=-1)
        # print (att_weight)
        # print (torch.sum(att_weight[0]))
        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def encode(self, src_sents_var, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
            dec_init_state, dec_init_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        src_token_embed = self.src_embed(src_sents_var)

        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        """Compute the initial decoder hidden state and cell state"""

        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, self.new_tensor(h_0.size()).zero_()

    def decode(self, src_encodings, src_sent_masks, dec_init_vec, batch):
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

        new_tensor = src_encodings.data.new
        batch_size = src_encodings.size(1)

        h_tm1 = dec_init_vec
        src_encodings = src_encodings.permute(1, 0, 2)
        src_seq_len = src_encodings.size(1)

        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # initialize the attentional vector
        att_tm1 = new_tensor(batch_size, self.hidden_size).zero_()
        assert att_tm1.requires_grad == False, "the att_tm1 requires grad is False"

        ast_seq_embed = self.ast_embed(ast_seq_var)

        if len(batch.variable_seq[0]) > 0:
            var_seq_embed = self.variable_embed(batch.variable_seq_var)

        scores = []
        variable_scores = []
        entity_scores = []
        att_probs = []

        entity_att_probs = []

        reg_loss_list = []
        hyp = Hypothesis()
        hyp.hidden_embedding_stack.append(h_tm1)
        hyp.v_hidden_embedding = h_tm1

        att_logits = []
        v_id = 0
        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(ast_seq_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim

            src_encodings_att_linear_cov = src_encodings_att_linear

            y_tm1_embed = y_tm1_embed.squeeze(0)

            # action instance
            current_node = action_seq[0][t]

            if self.use_att:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            if isinstance(current_node, ReduceAction):
                # non-terminal
                body_length = current_node.rule.body_length

                if self.use_input_lstm_encode:
                    input_lstm_embedding, (last_input_stack_state, last_input_stack_cell) = self.input_encode_lstm(
                        torch.stack(hyp.heads_embedding_stack[-body_length:]))
                    children_embedding = last_input_stack_state.squeeze(0)
                else:
                    children_embedding = torch.stack(hyp.heads_embedding_stack[-body_length:]).squeeze(1).mean(
                        dim=0).unsqueeze(0)
                x_point = self.input_stack_reduce_linear(torch.cat(
                    [children_embedding, x], 1))
                hyp.heads_embedding_stack = hyp.heads_embedding_stack[:-body_length]
                h_tm1_point = hyp.hidden_embedding_stack[-(body_length + 1)]

                hyp.hidden_embedding_stack = hyp.hidden_embedding_stack[:-body_length]
                hyp.heads_embedding_stack.append(x_point)



            elif isinstance(current_node, GenAction):
                # terminal
                x_point = x
                h_tm1_point = h_tm1
                hyp.heads_embedding_stack.append(x_point)
            else:
                raise ValueError

            if len(current_node.entities) > 0:
                h_tm1_e = h_tm1
                e_id = 0
                for ent in current_node.entities:
                    (h_e, cell_e), att_t_e, att_weight_e = self.step(x, h_tm1_e, src_encodings,
                                                                     src_encodings_att_linear_cov,
                                                                     src_sent_masks=src_sent_masks, lstm_type='entity')

                    score_e_t = self.readout_entity(att_t_e)
                    score_e = torch.softmax(score_e_t, dim=-1)
                    h_tm1_e = (h_e, cell_e)

                    entity_scores.append(torch.log(score_e))
                    entity_align_id = current_node.entity_align[e_id]
                    if not (entity_align_id == -1):
                        entity_att_probs.append(torch.log(att_weight_e[0, entity_align_id].sum()))

                    e_id += 1

            if len(current_node.variables) > 0:
                input_v = att_tm1
                for var in current_node.variables:
                    (h_t, cell_t), att_t, att_weight = self.step(input_v, hyp.v_hidden_embedding,
                                                                 src_encodings,
                                                                 src_encodings_att_linear,
                                                                 src_sent_masks=src_sent_masks,
                                                                 lstm_type='variable')
                    score_v_t = self.readout_variable(att_t)
                    score_v = torch.log_softmax(score_v_t, dim=-1)
                    variable_scores.append(score_v)
                    input_v = var_seq_embed[v_id]
                    v_id += 1
                    hyp.v_hidden_embedding = (h_t, cell_t)


            (h_t, cell_t), att_t, att_weight = self.step(x_point, h_tm1_point, src_encodings,
                                                         src_encodings_att_linear_cov,
                                                         src_sent_masks=src_sent_masks)

            if self.few_shot_mode == 'proto_train':
                action_proto = self.action_proto
            elif self.few_shot_mode == 'supervised_train':
                action_proto = self.action_embedding.weight
                action_proto = (action_proto.t() * self.action_mask).t()
            elif self.few_shot_mode == 'fine_tune':
                action_proto = self.action_embedding.weight

            score_t = torch.mm(action_proto, att_t.t())  # E.q. (6)


            score_t = score_t.t()
            if self.train_mask:
                if self.few_shot_mode == 'supervised_train':
                    score = masked_log_softmax(score_t, self.action_mask)
                elif self.few_shot_mode == 'proto_train':
                    score = masked_log_softmax(score_t, self.proto_mask)
                elif self.few_shot_mode == 'fine_tune':
                    score = torch.log_softmax(score_t, dim=-1)
            else:
                score = torch.log_softmax(score_t, dim=-1)

            scores.append(score)
            predicted_action_instance = action_seq[0][t + 1]

            ignore_items = set([SLOT_PREFIX, VAR_NAME, 'e'])

            if self.att_reg and (not (
                    len(predicted_action_instance.prototype_tokens) == 1 and (predicted_action_instance.prototype_tokens[0] in ignore_items))):

                attention_ratio = torch.nn.functional.sigmoid(self.att_reg_linear(att_t))

                pad_list = [0] * (src_seq_len - len(action_seq[0][t + 1].string_sim_score))

                cond_tensor = self.new_tensor(action_seq[0][t + 1].cond_score + pad_list).unsqueeze(0)

                strsim_tensor = self.new_tensor(action_seq[0][t + 1].string_sim_score + pad_list).unsqueeze(0)

                reg_weight = attention_ratio * cond_tensor + (
                        (1 - attention_ratio) * strsim_tensor)

                reg_weight = torch.div(reg_weight, reg_weight.sum())

                reg_loss = torch.abs(att_weight - reg_weight)
                reg_loss = reg_loss.sum()

                reg_loss_list.append(reg_loss)
            if self.sup_attention:
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_actions):
                        action_t = example.tgt_actions[t]
                        cand_src_tokens = AttentionUtil.get_candidate_tokens_to_attend(example.src_sent, action_t)
                        if cand_src_tokens:
                            att_prob = [torch.log(att_weight[e_id, token_id + 1].unsqueeze(0)) for token_id in
                                        cand_src_tokens]
                            if len(att_prob) > 1:
                                att_prob = torch.cat(att_prob).sum().unsqueeze(0)
                            else:
                                att_prob = att_prob[0]

                            att_probs.append(att_prob)
            hyp.hidden_embedding_stack.append((h_t, cell_t))
            att_logits.append(att_t)
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
        res_dict = {}

        res_dict['scores'] = (scores, entity_scores, variable_scores)
        res_dict['att_logits'] = att_logits
        if self.sup_attention:
            res_dict['att_probs'] = att_probs

        if self.att_reg:
            if len(reg_loss_list) > 0:
                reg_loss = torch.stack(reg_loss_list).sum()
                res_dict['reg_loss'] = reg_loss
            else:
                res_dict['reg_loss'] = -1

        if len(entity_att_probs) > 0:
            entity_loss = torch.stack(entity_att_probs).sum()
        else:
            entity_loss = -1
        res_dict['entity_loss'] = entity_loss
        return res_dict

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
        if self.training:
            sent_log_scores = -self.action_label_smoothing_layer(action_scores, action_var)
            if entity_scores is not None:
                entity_log_scores = -self.entity_label_smoothing_layer(entity_scores, entity_var)
            if variable_scores is not None:
                if len(self.variable_vocab) > 2:
                    variable_log_scores = -self.variable_label_smoothing_layer(variable_scores, variable_var)
                else:
                    variable_log_scores = torch.gather(variable_scores, -1, variable_var.unsqueeze(-1)).squeeze(-1)


        sent_log_scores = sent_log_scores * (1. - torch.eq(action_var, 0).float())  # 0 is pad

        sent_log_scores = sent_log_scores.sum(dim=0)

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

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None, lstm_type='decoder'):
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

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encodings_att_linear, mask=src_sent_masks)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        if self.dropout:
            att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t

    def forward(self, examples):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """
        src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        src_sents_var = src_batch.src_sents_var
        src_sents_len = src_batch.src_sents_len

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)
        loss = []
        loss_sup = []
        loss_reg = []
        loss_ent = []
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda)

            res_dict = self.decode(
                src_encodings[:, i, :].unsqueeze(1),
                src_sent_masks[i].unsqueeze(0), (
                    dec_init_vec[0][i, :].unsqueeze(0),
                    dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
            if self.sup_attention and res_dict['att_probs']:
                loss_sup.append(res_dict['att_probs'])

            loss_s = self.score_decoding_results(res_dict['scores'], tgt_batch)

            if self.att_reg:
                if isinstance(res_dict['reg_loss'], torch.Tensor):
                    loss_reg.append(self.att_reg * res_dict['reg_loss'])

            if isinstance(res_dict['entity_loss'], torch.Tensor):
                loss_ent.append(res_dict['entity_loss'])
            loss.append(loss_s)

        ret_val = [torch.stack(loss)]
        if self.sup_attention:
            ret_val.append(loss_sup)
        else:
            ret_val.append([])
        if self.att_reg:
            ret_val.append(loss_reg)
        else:
            ret_val.append([])

        ret_val.append(loss_ent)
        return ret_val

    def assign_nodes_back(self, node, padding, origin_node_list, c=0):

        if node.has_children():
            for child in node.children:
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
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode='token')
        entity_align = []

        for id, token in enumerate(example.src_sent):
            if is_lit(token, self.args.lang):
                entity_align.append(id)

        # TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, [src_sents_var.size(0)])
        src_encodings = src_encodings.permute(1, 0, 2)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        h_tm1 = self.init_decoder_state(last_state, last_cell)

        # tensor constructors
        new_float_tensor = src_encodings.data.new
        if self.use_cuda:
            new_long_tensor = torch.cuda.LongTensor
        else:
            new_long_tensor = torch.LongTensor

        att_tm1 = torch.zeros(1, self.hidden_size, requires_grad=True)
        hyp_scores = torch.zeros(1, requires_grad=True)
        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        node = RuleVertex('<s>')
        bos_id = self.vertex_vocab[node]
        vocab_size = len(self.action_vocab)

        first_hyp = Hypothesis()
        first_hyp.tgt_ids_stack.append(bos_id)

        first_hyp.hidden_embedding_stack.append(h_tm1)
        first_hyp.current_att_cov = torch.zeros(1, src_sents_var.size(0), requires_grad=False)
        first_hyp.var_id.append(1)
        first_hyp.ent_id.append(1)
        first_hyp.v_hidden_embedding = h_tm1
        if self.use_cuda:
            first_hyp.current_att_cov = first_hyp.current_att_cov.cuda()

        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step and len(hypotheses) > 0:

            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                src_encodings_att_linear.size(1),
                                                                                src_encodings_att_linear.size(2))

            y_tm1 = new_long_tensor([hyp.tgt_ids_stack[-1] for hyp in hypotheses])
            y_tm1_embed = self.ast_embed(y_tm1)

            if self.use_att:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            if not t == 0:
                x_list = []
                state_list = []
                cell_list = []
                state, cell = h_tm1
                for id, hyp in enumerate(hypotheses):
                    if hyp.heads_stack[-1].is_terminal():
                        hyp.heads_embedding_stack.append(x[id])
                        x_list.append(x[id])
                        state_list.append(state[id])
                        cell_list.append(cell[id])
                    else:
                        # non-terminal
                        body_length = hyp.actions[-1].rule.body_length

                        if self.use_input_lstm_encode:
                            input_lstm_embedding, (
                                last_input_stack_state, last_input_stack_cell) = self.input_encode_lstm(
                                torch.stack(hyp.heads_embedding_stack[-body_length:]).unsqueeze(1))
                            children_embedding = last_input_stack_state.squeeze(0)
                        else:
                            children_embedding = torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(
                                dim=0).unsqueeze(0)
                        x_point = self.input_stack_reduce_linear(torch.cat(
                            [children_embedding,
                             x[id].unsqueeze(0)], 1)).squeeze()
                        hyp.heads_embedding_stack = hyp.heads_embedding_stack[:-body_length]

                        x_list.append(x_point)
                        state_point, cell_point = hyp.hidden_embedding_stack[-(body_length + 1)]
                        state_list.append(state_point)
                        cell_list.append(cell_point)
                        hyp.hidden_embedding_stack = hyp.hidden_embedding_stack[:-body_length]
                        hyp.heads_embedding_stack.append(x_point)

                    if len(hyp.actions[-1].entities) > 0:
                        h_tm1_e = (h_tm1[0][id].unsqueeze(0), h_tm1[1][id].unsqueeze(0))
                        ent_list = []
                        for ent in hyp.actions[-1].entities:
                            (h_t_e, cell_t_e), att_t_e, att_weight_e = self.step(x[id].unsqueeze(0), h_tm1_e,
                                                                                 src_encodings,
                                                                                 src_encodings_att_linear,
                                                                                 src_sent_masks=None,
                                                                                 lstm_type='entity')

                            score_e_t = self.readout_entity(att_t_e)
                            score_e = torch.softmax(score_e_t, dim=-1)

                            score_e = score_e.log()
                            score_e[0][0] = -1e3
                            _, _ent_prev_word = score_e.max(1)
                            e_id = _ent_prev_word.item()
                            hyp.ent_id.append(e_id)
                            ent_list.append(self.entity_vocab.id2token[e_id])
                            h_tm1_e = (h_t_e, cell_t_e)

                        ent_node = hyp.heads_stack[-1] if isinstance(hyp.heads_stack[-1], RuleVertex) else \
                            hyp.heads_stack[
                                -1].vertex

                        if isinstance(hyp.actions[-1], ReduceAction):
                            c = 0
                            for child in ent_node.children:
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

                        assert c == len(hyp.actions[
                                            -1].entities), "variables {} and nodes variables {} num must be in consistency {} and {}".format(
                            c, len(hyp.actions[-1].entities), hyp.heads_stack[-1], hyp.actions[-1])

                    if len(hyp.actions[-1].variables) > 0:
                        var_list = []
                        input_v = att_tm1[id].unsqueeze(0)
                        for var in hyp.actions[-1].variables:

                            (h_t, cell_t), att_t, att_weight = self.step(input_v, hyp.v_hidden_embedding,
                                                                         src_encodings,
                                                                         src_encodings_att_linear,
                                                                         src_sent_masks=None,
                                                                         lstm_type='variable')
                            score_v_t = self.readout_variable(att_t)

                            score_v = torch.log_softmax(score_v_t, dim=-1)
                            score_v[0][0] = -1e3
                            _, _var_prev_word = score_v.max(1)
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

                x = torch.stack(x_list)
                h_tm1 = (torch.stack(state_list), torch.stack(cell_list))

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear,
                                                         src_sent_masks=None)

            if self.few_shot_mode == 'proto_train':
                action_proto = self.action_proto
            elif self.few_shot_mode == 'supervised_train':
                action_proto = self.action_embedding(self.new_long_tensor([i for i in range(len(self.action_vocab))]))
                action_proto = (action_proto.t() * self.action_mask).t()
            elif self.few_shot_mode == 'fine_tune':
                action_proto = self.action_embedding(self.new_long_tensor([i for i in range(len(self.action_vocab))]))

            score_t = torch.mm(action_proto, att_t.t()).t()


            p_t = torch.log_softmax(score_t, dim=-1)
            p_t[0][0]= -1e3

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
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
                if isinstance(action_instance, ReduceAction):
                    temp_hyp.reduce_action_count = temp_hyp.reduce_action_count + 1
                    predict_body_length = action_instance.rule.body_length

                    if t == 0 or temp_hyp.reduce_action_count > 10:
                        continue

                    if predict_body_length == FULL_STACK_LENGTH:
                        predict_body_length = len(temp_hyp.heads_stack)

                    if predict_body_length > len(temp_hyp.heads_stack):
                        continue

                    temp_hyp.actions.append(action_instance)
                    temp_hyp.heads_stack = temp_hyp.reduce_actions(action_instance)
                    if temp_hyp.completed():
                        temp_hyp.tgt_ids_stack.append(self.vertex_vocab[action_instance.rule.head.copy()])
                        temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))
                        temp_hyp.tgt_ids_stack = temp_hyp.tgt_ids_stack[1:]
                        temp_hyp.actions = temp_hyp.actions
                        temp_hyp.score = new_hyp_score
                        completed_hypotheses.append(temp_hyp)
                    else:
                        temp_hyp.tgt_ids_stack.append(self.vertex_vocab[action_instance.rule.head.copy()])
                        temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))
                        new_hypotheses.append(temp_hyp)
                        live_hyp_ids.append(prev_hyp_id)
                        new_hyp_scores.append(new_hyp_score)
                else:
                    temp_hyp.reduce_action_count = 0
                    if word_id == self.action_vocab[GenAction(RuleVertex('<gen_pad>'))] or word_id == self.action_vocab[
                        GenAction(RuleVertex('<s>'))]:
                        continue

                    temp_hyp.actions.append(action_instance)
                    temp_hyp.heads_stack.append(action_instance.vertex.copy())
                    temp_hyp.tgt_ids_stack.append(self.vertex_vocab[action_instance.vertex.copy()])
                    temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))

                    new_hypotheses.append(temp_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size or len(new_hypotheses) == 0:
                break

            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]
            att_w_tm1 = att_weight[live_hyp_ids]
            hyp_scores = new_float_tensor(new_hyp_scores)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
            t += 1
        if len(completed_hypotheses) == 0:

            dummy_hyp = Hypothesis()
            completed_hypotheses.append(dummy_hyp)
        else:
            completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        params = {
            'state_dict': self.state_dict(),
            'args': self.args,
            'vocab': self.vocab,
            'action': self.action_proto,
            'train_action': self.train_action_set,
            'action_mask': self.action_mask,
            "action_freq": self.action_freq
        }

        torch.save(params, path)

    @classmethod
    def load(cls, model_path, use_cuda=False, loaded_vocab=None, args = None):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']

        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.dropout = args.dropout
        saved_args.att_reg = args.att_reg
        saved_args.lr = args.lr
        saved_args.label_smoothing = args.label_smoothing
        saved_args.sup_attention = args.sup_attention
        saved_args.hyp_embed = args.hyp_embed
        parser = cls(vocab, saved_args)
        if not 'entity_linear.weight' in saved_state:
            parser.entity_linear = None

        parser.load_state_dict(saved_state)
        parser.action_proto.copy_(params['action'])
        parser.train_action_set = params['train_action']
        parser.action_mask.copy_(params['action_mask'])
        src_vocab = None
        if loaded_vocab:
            src_vocab = update_vocab_related_parameters(parser, loaded_vocab)

        if use_cuda:
            parser = parser.cuda()

        return parser, src_vocab
