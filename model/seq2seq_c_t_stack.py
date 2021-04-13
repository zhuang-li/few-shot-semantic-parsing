# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from components.dataset import Batch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from common.registerable import Registrable
from model import nn_utils
from grammar.consts import *
from grammar.vertex import *
from grammar.rule import ReduceAction, GenAction, Rule
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
from model.locked_dropout import LockedDropout
from model.attention_util import AttentionUtil, Attention
import os


@Registrable.register('seq2seq_c_t_stack')
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

        self.src_embed = nn.Embedding(len(self.src_vocab), self.embed_size)

        self.ast_embed = nn.Embedding(len(self.vertex_vocab), self.action_embed_size)

        # general action embed
        self.general_action_vocab = vocab.general_action
        self.general_action_bce_loss = nn.BCEWithLogitsLoss()

        # whether to use att
        self.use_att = args.use_att
        if args.use_att:
            self.decoder_size = self.action_embed_size + self.hidden_size
        else:
            self.decoder_size = self.action_embed_size
        self.lstm = args.lstm
        if args.lstm == 'lstm':
            self.encoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)

            self.decoder_lstm = nn.LSTMCell(self.decoder_size, self.hidden_size)

            self.use_input_lstm_encode = args.use_input_lstm_encode
            if self.use_input_lstm_encode:
                #print (self.action_embed_size + self.hidden_size)
                self.input_encode_lstm = nn.LSTM(self.decoder_size, self.decoder_size, bidirectional=False)

            self.use_children_lstm_encode = args.use_children_lstm_encode
            if self.use_children_lstm_encode:
                #print (self.action_embed_size + self.hidden_size)
                self.children_encode_lstm = nn.LSTM(self.decoder_size, self.decoder_size, bidirectional=False)
        elif args.lstm == 'gru':
            self.encoder_lstm = nn.GRU(self.embed_size, self.hidden_size, bidirectional=True)

            self.decoder_lstm = nn.GRUCell(self.decoder_size, self.hidden_size)

            self.use_input_lstm_encode = args.use_input_lstm_encode
            if self.use_input_lstm_encode:
                # print (self.action_embed_size + self.hidden_size)
                self.input_encode_lstm = nn.GRU(self.decoder_size, self.decoder_size, bidirectional=False)

            self.use_children_lstm_encode = args.use_children_lstm_encode
            if self.use_children_lstm_encode:
                # print (self.action_embed_size + self.hidden_size)
                self.children_encode_lstm = nn.GRU(self.decoder_size, self.decoder_size, bidirectional=False)

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

        self.dropout_i = args.dropout_i
        if self.dropout_i > 0:
            self.dropout_i = LockedDropout(self.dropout_i)

        # prediction layer of the generation actions
        self.gen_vocab = vocab.gen_action
        self.readout_gen = nn.Linear(self.hidden_size, len(self.gen_vocab), bias=False)

        # prediction layer of the reduce actions
        self.re_vocab = vocab.re_action
        re_vocab_mask = nn_utils.length_array_to_mask_tensor(self.re_vocab.body_len_length_array,
                                                             use_cuda=args.use_cuda, valid_entry_has_mask_one=True)
        self.readout_re = nn_utils.BatchLinear(re_vocab_mask.size(0), re_vocab_mask.size(1), self.hidden_size,
                                               re_vocab_mask)

        # project the input stack and reduce head into the embedding space
        self.input_stack_reduce_linear = nn.Linear(self.decoder_size * 2, self.decoder_size,
                                                   bias=False)

        self.output_stack_reduce_linear = nn.Linear(self.decoder_size + self.hidden_size, self.hidden_size, bias=False)
        # project the attention vector stack and reduce head into the embedding space
        # self.attention_stack_reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # project the hidden vector stack and reduce head into the embedding space
        self.general_action_linear = nn.Linear(self.hidden_size, 1, bias=False)

        self.label_smoothing = None

        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.gen_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.gen_vocab),
                                                                     ignore_indices=[0, 1])

            self.re_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.re_vocab),
                                                                    ignore_indices=[0])

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

        self.attention = args.attention
        self.att_module = Attention(1, self.hidden_size, self.attention)

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        :return:
        """
        # (batch_size, src_sent_len)
        att_weight = self.att_module(h_t, src_encoding_att_linear, mask)
        #print (att_weight.size())
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
        # print (src_sents_var.size())
        src_token_embed = self.src_embed(src_sents_var)
        if self.dropout_i:
            src_token_embed = self.dropout_i(src_token_embed)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # print (last_state.size())
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
        # (batch_size, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        # print (src_encodings.size())
        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # initialize the attentional vector
        att_tm1 = new_tensor(batch_size, self.hidden_size).zero_()
        assert att_tm1.requires_grad == False, "the att_tm1 requires grad is False"
        # (batch_size, src_sent_len)

        # print (src_sent_masks.size())
        # (tgt_sent_len, batch_size, embed_size)
        ast_seq_embed = self.ast_embed(ast_seq_var)

        gen_scores = []
        re_scores = []
        att_probs = []
        hyp = Hypothesis()
        hyp.hidden_embedding_stack.append(h_tm1)

        #hyp.embedding_stack.append(att_tm1)
        att_logits = []

        cov_v = torch.zeros(1, len(action_seq), requires_grad=False)
        att_weight = torch.zeros(1, len(action_seq), requires_grad=False)
        #print (cov_v.size())
        if self.use_cuda:
            cov_v = cov_v.cuda()
            att_weight = att_weight.cuda()

        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(ast_seq_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim

            if self.use_coverage:
                #print (cov_v.unsqueeze(-1).size())
                #print (src_encodings_att_linear.clone().size())
                cov_v = cov_v + att_weight
                #print (torch.sigmoid(self.fertility(h_tm1[0][0])))
                src_encodings_att_linear_cov = src_encodings_att_linear.clone() + self.cov_linear(torch.div(cov_v.unsqueeze(-1), torch.sigmoid(self.fertility(h_tm1[0][0]))))
            else:
                src_encodings_att_linear_cov = src_encodings_att_linear

            y_tm1_embed = y_tm1_embed.squeeze(0)

            # action instance
            current_node = action_seq[0][t]
            predict_action_instance = action_seq[0][t + 1]

            if self.dropout_i:
                y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)

            if self.use_att:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed


            if isinstance(current_node, ReduceAction):
                # non-terminal
                body_length = current_node.rule.body_length
                """
                if body_length == FULL_STACK_LENGTH:
                    body_length = len(hyp.hidden_embedding_stack) - 1
                """
                # print (torch.stack(input_stack[:-body_length]).squeeze(1).mean(dim=0).unsqueeze(0).size())
                if self.use_input_lstm_encode:
                    input_lstm_embedding, (last_input_stack_state, last_input_stack_cell) = self.input_encode_lstm(
                        torch.stack(hyp.heads_embedding_stack[-body_length:]))
                    children_embedding = last_input_stack_state.squeeze(0)
                    #print(last_input_stack_state.size())
                else:
                    children_embedding = torch.stack(hyp.heads_embedding_stack[-body_length:]).squeeze(1).mean(
                        dim=0).unsqueeze(0)
                    #print (children_embedding.size())
                x = self.input_stack_reduce_linear(torch.cat(
                    [children_embedding, x], 1))
                hyp.heads_embedding_stack = hyp.heads_embedding_stack[:-body_length]
                h_tm1 = hyp.hidden_embedding_stack[-(body_length + 1)]
                hyp.hidden_embedding_stack = hyp.hidden_embedding_stack[:-body_length]
                hyp.heads_embedding_stack.append(x)
            elif isinstance(current_node, GenAction):
                # terminal
                hyp.heads_embedding_stack.append(x)
            else:
                raise ValueError

            #if self.dropout_i:
                #x = self.dropout_i(x.unsqueeze(0)).squeeze(0)

            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, src_encodings, src_encodings_att_linear_cov,
                                                         src_sent_masks=src_sent_masks)
            #print (t)
            #print (cov_v)
            if isinstance(predict_action_instance, ReduceAction):
                # (batch_size, tgt_vocab_size)
                temp_rep = []
                temp_embedding_stack = hyp.heads_embedding_stack[-(len(hyp.heads_embedding_stack)-1):]
                current_att = att_t.squeeze()
                for body_len in self.re_vocab.body_len_list:
                    # print (torch.stack(att_stack[-body_len:]).squeeze(1).mean(dim=0).size())
                    # print (torch.stack(att_stack[-body_len:]).squeeze(1).size())
                    #if body_len == FULL_STACK_LENGTH:
                        #body_len = len(hyp.heads_embedding_stack) - 1
                    if body_len <= len(temp_embedding_stack) or body_len == FULL_STACK_LENGTH:
                        if self.use_children_lstm_encode:
                            #print (torch.stack(temp_embedding_stack[-body_len:]).size())
                            lstm_embedding, (last_stack_state, last_stack_cell) = self.children_encode_lstm(
                                torch.stack(temp_embedding_stack[-body_len:]))
                            temp_rep.append(
                                self.output_stack_reduce_linear(torch.cat([last_stack_state.squeeze(), current_att])))
                            # print (last_stack_state.squeeze().size())
                        else:
                            #print (torch.stack(temp_embedding_stack[-body_len:]).squeeze(1).mean(dim=0).size())
                            temp_rep.append(self.output_stack_reduce_linear(torch.cat(
                                [torch.stack(temp_embedding_stack[-body_len:]).squeeze(1).mean(dim=0), current_att])))
                    else:
                        padding_tensor = torch.zeros(body_len - len(temp_embedding_stack), self.decoder_size,
                                                     dtype=torch.float)
                        if self.args.use_cuda:
                            padding_tensor = padding_tensor.cuda()
                        if self.use_children_lstm_encode:
                            # print (torch.cat([padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)],dim=0).unsqueeze(1).size())
                            lstm_embedding, (last_stack_state, last_stack_cell) = self.children_encode_lstm(
                                torch.cat([padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)],
                                          dim=0).unsqueeze(1))
                            temp_rep.append(
                                self.output_stack_reduce_linear(torch.cat([last_stack_state.squeeze(), current_att])))
                        else:
                            #print (padding_tensor.size())
                            #print (torch.stack(temp_embedding_stack[-body_len:]).squeeze(1).size())

                            temp_rep.append(self.output_stack_reduce_linear(torch.cat([torch.cat(
                                [padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)], dim=0).mean(
                                dim=0), current_att])))

                temp_rep = torch.stack(temp_rep)
                temp_rep = torch.tanh(temp_rep)
                re_score_t = self.readout_re(temp_rep)  # E.q. (6)
                pad_dimension_elem = torch.zeros(1).fill_(-float('inf'))
                if self.args.use_cuda:
                    pad_dimension_elem = pad_dimension_elem.cuda()
                re_score_t = torch.cat([pad_dimension_elem, re_score_t], -1)
                # print (re_score_t)
                re_score = torch.log_softmax(re_score_t, dim=-1)
                # print (re_score)
                re_scores.append(re_score)
                #predict_body_length = predict_action_instance.rule.body_length
                # print ("before")
                # print(len(hyp.embedding_stack))
                # print ("length", predict_body_length)

                #next_att = att_t
                #temp_rep[self.re_vocab.body_len_list.index(predict_body_length)].unsqueeze(0)

                #if predict_body_length == FULL_STACK_LENGTH:
                    #predict_body_length = len(hyp.embedding_stack)

                #hyp.embedding_stack = hyp.embedding_stack[:-predict_body_length]
                #print (next_att.size())
                #hyp.embedding_stack.append(next_att)
            elif isinstance(predict_action_instance, GenAction):
                # (batch_size, tgt_vocab_size)
                gen_score_t = self.readout_gen(att_t)  # E.q. (6)
                gen_score = torch.log_softmax(gen_score_t, dim=-1)
                gen_scores.append(gen_score)
                # stack state
                #next_att = att_t
                #hyp.embedding_stack.append(next_att)

            else:
                raise ValueError
            #cov_v = cov_v + att_weight
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
            att_logits.append(att_t)
            hyp.hidden_embedding_stack.append((h_t, cell_t))

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        gen_scores = torch.stack(gen_scores)
        re_scores = torch.stack(re_scores).unsqueeze(1)
        att_logits = torch.stack(att_logits)
        if len(att_probs) > 0:
            att_probs = torch.stack(att_probs).sum()
        else:
            att_probs = None
        #print (att_probs.size())
        if self.sup_attention:
            return (gen_scores, re_scores), att_logits, att_probs
        else:
            return (gen_scores, re_scores), att_logits

    def score_decoding_results(self, scores, att_logits, batch):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        # (tgt_sent_len, batch_size, tgt_vocab_size)
        (gen_scores, re_scores) = scores
        gen_action_var = batch.hierarchy_action_seq_var[GenAction.__name__]
        re_action_var = batch.hierarchy_action_seq_var[ReduceAction.__name__]

        general_action_seq_var_omitted = batch.general_action_seq_var[1:].float()
        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            gen_sent_log_scores = -self.gen_label_smoothing_layer(gen_scores, gen_action_var)
            re_sent_log_scores = -self.re_label_smoothing_layer(re_scores, re_action_var)
        else:
            # (tgt_sent_len, batch_size)
            gen_sent_log_scores = torch.gather(gen_scores, -1, gen_action_var.unsqueeze(-1)).squeeze(-1)
            re_sent_log_scores = torch.gather(re_scores, -1, re_action_var.unsqueeze(-1)).squeeze(-1)

        gen_sent_log_scores = gen_sent_log_scores * (1. - torch.eq(gen_action_var, 0).float())  # 0 is pad

        re_sent_log_scores = re_sent_log_scores * (1. - torch.eq(re_action_var, 0).float())  # 0 is pad
        # print (gen_sent_log_scores.size())

        # (batch_size)
        # print (gen_sent_log_scores.size())
        gen_sent_log_scores = gen_sent_log_scores.sum(dim=0)
        # print (gen_sent_log_scores)
        # print (gen_sent_log_scores)
        # (batch_size)
        re_sent_log_scores = re_sent_log_scores.sum(dim=0)
        # print (att_logits.size())
        # print (self.general_action_linear(att_logits.squeeze()).squeeze().size())
        # print (general_action_seq_var_omitted.squeeze().size())
        general_action_loss = self.general_action_bce_loss(self.general_action_linear(att_logits.squeeze()).squeeze(),
                                                           general_action_seq_var_omitted.squeeze())

        # generate action loss
        #print (gen_sent_log_scores)
        gen_loss = -gen_sent_log_scores.unsqueeze(0)[0]
        # reduce action loss
        re_loss = -re_sent_log_scores.unsqueeze(0)[0]
        # print (general_action_loss)
        # print (gen_loss)
        # print (re_loss)
        return general_action_loss + gen_loss + re_loss

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encodings_att_linear, mask=src_sent_masks)
        # print (src_sent_masks)
        # vector for action prediction
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
        # src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        src_sents_var = src_batch.src_sents_var
        src_sents_len = src_batch.src_sents_len
        # action_seq_var = batch.action_seq_var

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)
        loss = []
        loss_sup = []
        #print (len(src_batch))
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda)
            if self.sup_attention:
                (gen_scores, re_scores), action_token_logits, att_prob = self.decode(
                    src_encodings[:, i, :].unsqueeze(1),
                    src_sent_masks[i].unsqueeze(0), (
                        dec_init_vec[0][i, :].unsqueeze(0),
                        dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
                if att_prob:
                    loss_sup.append(att_prob)
            else:
                (gen_scores, re_scores), action_token_logits = self.decode(src_encodings[:, i, :].unsqueeze(1),
                                                                           src_sent_masks[i].unsqueeze(0), (
                                                                               dec_init_vec[0][i, :].unsqueeze(0),
                                                                               dec_init_vec[1][i, :].unsqueeze(0)),
                                                                           tgt_batch)
            loss_s = self.score_decoding_results((gen_scores, re_scores), action_token_logits, tgt_batch)
            loss.append(loss_s)
        ret_val = [torch.stack(loss)]
        if self.sup_attention:
            ret_val.append(loss_sup)
        # print(loss.data)
        return ret_val

    def beam_search(self, example, decode_max_time_step, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode='token')

        """
        print ("===============================preview===================================")
        print(" ".join(example.src_sent))
        print(example.tgt_code_no_var_str)
        print ("===============================preview====================================")
        """
        # TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, [src_sents_var.size(0)])
        # (1, query_len, hidden_size * 2)
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
        att_w_tm1 = torch.zeros(1, src_sents_var.size(0), requires_grad=True)
        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
            att_w_tm1 = att_w_tm1.cuda()
        # todo change it back
        # eos_id = self.action_vocab['</s>']
        node = RuleVertex('<s>')
        bos_id = self.vertex_vocab[node]
        re_vocab_size = len(self.re_vocab)
        gen_vocab_size = len(self.gen_vocab)

        first_hyp = Hypothesis()
        first_hyp.tgt_ids_stack.append(bos_id)
        #first_hyp.embedding_stack.append(att_tm1)
        first_hyp.hidden_embedding_stack.append(h_tm1)
        first_hyp.current_att_cov = torch.zeros(1, src_sents_var.size(0), requires_grad=False)
        if self.use_cuda:
            first_hyp.current_att_cov = first_hyp.current_att_cov.cuda()
            # hypotheses = [[bos_id]]
        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step and len(hypotheses) > 0:
            # if t == 50:
            # print (t)
            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            if self.use_coverage:
                expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                src_encodings_att_linear.size(1),
                                                                                src_encodings_att_linear.size(2)).clone()

                for id, hyp in enumerate(hypotheses):
                    hyp.current_att_cov = hyp.current_att_cov + att_w_tm1[id].view(hyp.current_att_cov.size())
                    #print (t)
                    #print (hyp.current_att_cov)
                    expanded_src_encodings_att_linear[id] = expanded_src_encodings_att_linear[id] + self.cov_linear(torch.div(hyp.current_att_cov.unsqueeze(-1), torch.sigmoid(self.fertility(h_tm1[id][0]))))

            else:
                expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                    src_encodings_att_linear.size(1),
                                                                                    src_encodings_att_linear.size(2))


            y_tm1 = new_long_tensor([hyp.tgt_ids_stack[-1] for hyp in hypotheses])
            y_tm1_embed = self.ast_embed(y_tm1)

            if self.dropout_i:
                y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)

            if self.use_att:
                #print (hyp_num)
                #print (y_tm1_embed.size())
                #print (att_tm1.size())
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            # assert not torch.isnan(x).any(), "x has no nan"
            if not t == 0:
                x_list = []
                state_list = []
                cell_list = []
                for id, hyp in enumerate(hypotheses):
                    state, cell = h_tm1
                    if hyp.heads_stack[-1].is_terminal():
                        hyp.heads_embedding_stack.append(x[id])
                        x_list.append(x[id])
                        state_list.append(state[id])
                        cell_list.append(cell[id])
                        """
                        if self.use_coverage:
                            hyp.current_att_cov = hyp.current_att_cov + att_w_tm1[id].view(hyp.current_att_cov.size())

                            expanded_src_encodings_att_linear[id] = expanded_src_encodings_att_linear[id] + self.cov_linear(
                                hyp.current_att_cov.unsqueeze(-1))
                        """
                    else:
                        # non-terminal
                        body_length = hyp.actions[-1].rule.body_length
                        # print (torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(dim=0))
                        # print (torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(dim=0).unsqueeze(0).size())
                        # print (x[id].size())

                        if self.use_input_lstm_encode:
                            #print (torch.stack(hyp.heads_embedding_stack[-body_length:]).size())
                            input_lstm_embedding, (
                            last_input_stack_state, last_input_stack_cell) = self.input_encode_lstm(
                                torch.stack(hyp.heads_embedding_stack[-body_length:]).unsqueeze(1))
                            children_embedding = last_input_stack_state.squeeze(0)
                            #print (last_input_stack_state.size())
                        else:
                            children_embedding = torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(
                                dim=0).unsqueeze(0)
                            #print (children_embedding.size())
                        x_point = self.input_stack_reduce_linear(torch.cat(
                            [children_embedding,
                             x[id].unsqueeze(0)], 1)).squeeze()
                        hyp.heads_embedding_stack = hyp.heads_embedding_stack[:-body_length]
                        # assert not torch.isnan(x_point).any(), "x has no nan"
                        # print (x_point.size())
                        x_list.append(x_point)
                        try:
                            state_point, cell_point = hyp.hidden_embedding_stack[-(body_length + 1)]
                        except IndexError:
                            print (len(hyp.hidden_embedding_stack))
                            print (body_length)
                            print (hyp.heads_stack[-1])
                            print (hyp.heads_stack)
                        state_list.append(state_point)
                        cell_list.append(cell_point)
                        hyp.hidden_embedding_stack = hyp.hidden_embedding_stack[:-body_length]
                        hyp.heads_embedding_stack.append(x_point)
                x = torch.stack(x_list)
                h_tm1 = (torch.stack(state_list), torch.stack(cell_list))

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear,
                                                         src_sent_masks=None)
            """
            src_sent = ['</s>'] + example.src_sent + ['<s/>']
            print ("==================================================")
            print(src_sent)

            print(att_weight)
            print (torch.argmax(att_weight,dim=1).item())
            if torch.argmax(att_weight,dim=1).item()< len(src_sent) + 2:
                print (src_sent[torch.argmax(att_weight,dim=1).item()])
            """
            if self.general_action_vocab.token2id[ReduceAction.__name__] == 1:
                re_bin_score = torch.log(torch.sigmoid(self.general_action_linear(att_t)))
                gen_bin_score = torch.log(1 - torch.sigmoid(self.general_action_linear(att_t)))
            else:
                gen_bin_score = torch.log(torch.sigmoid(self.general_action_linear(att_t)))
                re_bin_score = torch.log(1 - torch.sigmoid(self.general_action_linear(att_t)))

            p_t = []
            for id, hyp in enumerate(hypotheses):
                # re_score
                temp_rep = []
                current_att_t = att_t[id].squeeze()
                if not t == 0:
                    temp_embedding_stack = hyp.heads_embedding_stack
                    for body_len in self.re_vocab.body_len_list:
                        # print (torch.stack(att_stack[-body_len:]).squeeze(1).mean(dim=0).size())
                        # print (torch.stack(att_stack[-body_len:]).squeeze(1).size())
                        # print (temp_embedding_stack)

                        if body_len <= len(temp_embedding_stack) or body_len == FULL_STACK_LENGTH:
                            # print (torch.stack(temp_embedding_stack[-body_len:]).squeeze(1).mean(dim=0).size())
                            if self.use_children_lstm_encode:
                                # print (torch.stack(temp_embedding_stack[-body_len:]).size())
                                lstm_embedding, (last_stack_state, last_stack_cell) = self.children_encode_lstm(
                                    torch.stack(temp_embedding_stack[-body_len:]).unsqueeze(1))
                                temp_rep.append(
                                    self.output_stack_reduce_linear(torch.cat([last_stack_state.squeeze(), current_att_t])))
                            else:
                                #temp_rep.append(torch.stack(temp_embedding_stack[-body_len:]).squeeze(1).mean(dim=0))

                                temp_rep.append(self.output_stack_reduce_linear(torch.cat(
                                    [torch.stack(temp_embedding_stack[-body_len:]).squeeze(1).mean(dim=0), current_att_t])))

                        else:
                            padding_tensor = torch.zeros(body_len - len(temp_embedding_stack), self.decoder_size,
                                                         dtype=torch.float)
                            if self.args.use_cuda:
                                padding_tensor = padding_tensor.cuda()
                            if self.use_children_lstm_encode:
                                # print (torch.cat([padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)],dim=0).unsqueeze(1).size())
                                lstm_embedding, (last_stack_state, last_stack_cell) = self.children_encode_lstm(
                                    torch.cat([padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)],
                                              dim=0).unsqueeze(1))
                                #temp_rep.append(last_stack_state.squeeze())
                                temp_rep.append(
                                    self.output_stack_reduce_linear(torch.cat([last_stack_state.squeeze(), current_att_t])))
                            else:
                                # print (torch.cat([padding_tensor, torch.stack(att_stack[-body_len:]).squeeze(1)],dim=0).size())
                                #temp_rep.append(
                                 #   torch.cat([padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)],
                                  #            dim=0).mean(dim=0))
                                temp_rep.append(self.output_stack_reduce_linear(torch.cat([torch.cat(
                                    [padding_tensor, torch.stack(temp_embedding_stack[-body_len:]).squeeze(1)], dim=0).mean(
                                    dim=0), current_att_t])))

                    temp_rep = torch.stack(temp_rep)
                    temp_rep = torch.tanh(temp_rep)
                    #hyp.current_re_emb = temp_rep
                    # print (temp_rep.size())
                    re_score_t = self.readout_re(temp_rep)  # E.q. (6)
                    pad_dimension_elem = torch.zeros(1).fill_(-float('inf'))
                    if self.args.use_cuda:
                        pad_dimension_elem = pad_dimension_elem.cuda()
                    re_score_t = torch.cat([pad_dimension_elem, re_score_t], -1)
                    # print (re_score_t)
                    re_score = torch.log_softmax(re_score_t, dim=-1)
                    re_score[0] = -1000.0
                else:
                    re_score = torch.zeros(len(self.re_vocab)).fill_(-1000.0)
                    if self.use_cuda:
                        re_score = re_score.cuda()
                # smoothing
                # print (re_score)
                # print (re_score.size())
                # print (re_score)
                # p_t.append(re_score + re_bin_score[id])

                gen_score_t = self.readout_gen(current_att_t)  # E.q. (6)

                #hyp.current_gen_emb = current_att_t

                gen_score = torch.log_softmax(gen_score_t, dim=-1)
                #print (gen_score.size())
                #print (re_score.size())
                re_gen_cat = torch.cat([re_score + re_bin_score[id], gen_score + gen_bin_score[id]], dim=0)
                # print (re_gen_cat.size())
                p_t.append(re_gen_cat)

            p_t = torch.stack(p_t)
            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            # print(new_hyp_scores)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos // (re_vocab_size + gen_vocab_size)
            word_ids = top_new_hyp_pos % (re_vocab_size + gen_vocab_size)

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data.tolist(),
                                                           word_ids.cpu().data.tolist(),
                                                           top_new_hyp_scores.cpu().data.tolist()):
                try:
                    temp_hyp = hypotheses[prev_hyp_id].copy()
                except IndexError:
                    print(len(hypotheses))
                    print(hypotheses)
                    print(prev_hyp_id)
                    print(re_score_t)
                    print(re_score)
                    print(new_hyp_scores)

                if word_id < re_vocab_size:
                    # print (self.re_vocab[ReduceAction(Rule(RuleVertex('<re_pad>')))])
                    # print (temp_hyp.reduce_action_count)
                    temp_hyp.reduce_action_count = temp_hyp.reduce_action_count + 1
                    re_action_instance = self.re_vocab.id2token[word_id]
                    predict_body_length = re_action_instance.rule.body_length
                    # print (self.re_vocab.body_len_list.index(predict_body_length))
                    # print (self.re_vocab.body_len_list)
                    # print(predict_body_length)
                    if word_id == self.re_vocab[
                        ReduceAction(Rule(RuleVertex('<re_pad>')))] or t == 0 or temp_hyp.reduce_action_count > 10:
                        #print ("continue")
                        continue

                    #temp_hyp.current_att = temp_hyp.current_re_emb[
                        #self.re_vocab.body_len_list.index(predict_body_length)].view(temp_hyp.current_gen_emb.size())

                    # print (temp_hyp.current_att.size())
                    if predict_body_length == FULL_STACK_LENGTH:
                        predict_body_length = len(temp_hyp.heads_stack)

                    if predict_body_length > len(temp_hyp.heads_stack):
                        continue
                    # print (prev_hyp_id)
                    # print (len(temp_hyp.embedding_stack))
                    temp_hyp.actions.append(re_action_instance)
                    temp_hyp.heads_stack = temp_hyp.reduce_actions(re_action_instance)
                    if temp_hyp.completed():

                        temp_hyp.tgt_ids_stack.append(self.vertex_vocab[re_action_instance.rule.head])
                        # temp_hyp.actions.append(re_action_instance)

                        #temp_hyp.embedding_stack = temp_hyp.embedding_stack[:-predict_body_length]
                        #temp_hyp.embedding_stack.append(temp_hyp.current_att)
                        temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))

                        temp_hyp.tgt_ids_stack = temp_hyp.tgt_ids_stack[1:]
                        temp_hyp.actions = temp_hyp.actions
                        temp_hyp.score = new_hyp_score
                        #new_hypotheses.append(temp_hyp)
                        completed_hypotheses.append(temp_hyp)
                    else:
                        temp_hyp.tgt_ids_stack.append(self.vertex_vocab[re_action_instance.rule.head])
                        # temp_hyp.actions.append(re_action_instance)

                        #temp_hyp.embedding_stack = temp_hyp.embedding_stack[:-predict_body_length]
                        #temp_hyp.embedding_stack.append(temp_hyp.current_att)
                        temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))
                        new_hypotheses.append(temp_hyp)
                        live_hyp_ids.append(prev_hyp_id)
                        new_hyp_scores.append(new_hyp_score)
                else:
                    temp_hyp.reduce_action_count = 0
                    word_id = word_id - re_vocab_size
                    if word_id == self.gen_vocab[GenAction(RuleVertex('<gen_pad>'))] or word_id == self.gen_vocab[
                        GenAction(RuleVertex('<s>'))]:
                        #print ("continue")
                        continue

                    gen_action_instance = self.gen_vocab.id2token[word_id]
                    temp_hyp.actions.append(gen_action_instance)
                    temp_hyp.heads_stack.append(gen_action_instance.vertex)
                    temp_hyp.tgt_ids_stack.append(self.vertex_vocab[gen_action_instance.vertex])
                    temp_hyp.hidden_embedding_stack.append((h_t[prev_hyp_id], cell_t[prev_hyp_id]))
                    #temp_hyp.current_att = temp_hyp.current_gen_emb
                    #temp_hyp.embedding_stack.append(temp_hyp.current_att)
                    new_hypotheses.append(temp_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)
                """
                assert len(temp_hyp.heads_stack) == len(
                    temp_hyp.embedding_stack) - 1, "the length of heads stack {} must equal to the length of the embedding stack {}".format(
                    len(temp_hyp.heads_stack), len(
                        temp_hyp.embedding_stack) - 1)
                
                assert len(temp_hyp.heads_stack) == len(
                    temp_hyp.hidden_embedding_stack) - 1, "the length of heads stack must equal to the length of the hidden embedding stack"
                assert len(temp_hyp.hidden_embedding_stack) == len(
                    temp_hyp.embedding_stack), "the length of hidden embedding stack must equal to the length of the embedding stack"
                """
            #print(new_hypotheses[0].actions)
            #print(new_hypotheses[0].actions[-1])
            if len(completed_hypotheses) == beam_size or len(new_hypotheses) == 0:
                break

            # live_hyp_ids = new_long_tensor(live_hyp_ids)
            """
            att_tm1_list = []
            for hyp in new_hypotheses:
                att_tm1_list.append(hyp.current_att)
            """
            # for emb in att_tm1_list:
            # print (emb.size())

            #att_tm1 = torch.stack(att_tm1_list)
            # print (att_tm1.size())
            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]
            att_w_tm1 = att_weight[live_hyp_ids]
            hyp_scores = new_float_tensor(new_hyp_scores)  # new_hyp_scores[live_hyp_ids]
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
    def load(cls, model_path, use_cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.use_cuda = use_cuda
        parser = cls(vocab, saved_args)

        parser.load_state_dict(saved_state)

        if use_cuda: parser = parser.cuda()
        parser.eval()

        return parser
