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
from model.attention_util import AttentionUtil, dot_prod_attention
import os


@Registrable.register('seq2seq_c_t')
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
        self.action_vocab = vocab.action

        self.src_embed = nn.Embedding(len(self.src_vocab), self.embed_size)

        self.action_embed = nn.Embedding(len(self.action_vocab), self.action_embed_size)

        # general action embed
        self.use_general_action_embed = args.use_general_action_embed
        if self.use_general_action_embed:
            self.general_action_embed_size = args.general_action_embed_size
            self.general_action_vocab = vocab.general_action
            self.general_action_embed = nn.Embedding(len(self.general_action_vocab), self.general_action_embed_size)
            decoder_size = self.action_embed_size + self.general_action_embed_size + self.hidden_size
        else:
            decoder_size = self.action_embed_size + self.hidden_size

        self.encoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)

        self.decoder_lstm = nn.LSTMCell(decoder_size, self.hidden_size)

        self.use_children_lstm_encode = args.use_children_lstm_encode
        if self.use_children_lstm_encode:
            self.children_encode_lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size, bias=False)

        # proto type layer
        self.prototype_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size, bias=False)
        self.bce_loss = nn.BCEWithLogitsLoss()

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(self.hidden_size, len(self.action_vocab), bias=False)

        # supervised attention
        self.sup_attention = args.sup_attention

        # dropout layer
        self.dropout = args.dropout
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout)

        self.dropout_i = args.dropout_i
        if self.dropout_i > 0:
            self.dropout_i = LockedDropout(self.dropout_i)

        self.label_smoothing = None

        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.action_vocab),
                                                                 ignore_indices=[0])

        self.mem_net = args.mem_net

        # hierarchy classification
        self.hierarchy_label = args.hierarchy_label
        if self.hierarchy_label == 'base':
            self.general_action_vocab = vocab.general_action
            # prediction layer of the base label vocabulary
            self.readout_b = nn.Linear(self.hidden_size, len(self.general_action_vocab), bias=False)
            for id, index_list in self.general_action_vocab.action_hierarchy.items():
                if self.use_cuda:
                    self.general_action_vocab.action_hierarchy[id] = torch.LongTensor(index_list).cuda()
                else:
                    self.general_action_vocab.action_hierarchy[id] = torch.LongTensor(index_list)
                # print (torch.Tensor(index_list))

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

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        dec_init_cell = self.decoder_cell_init(enc_last_cell)
        dec_init_state = torch.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

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

        action_seq_var = batch.action_seq_var
        general_action_seq_var = batch.general_action_seq_var
        action_seq = batch.action_seq

        assert action_seq_var.size(0) == (len(
            action_seq[0]) + 1), "length of sequence variable should be the same with the length of the action sequence"

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
        action_token_embed = self.action_embed(action_seq_var)
        if self.use_general_action_embed:
            general_action_token_embed = self.general_action_embed(general_action_seq_var)

        scores = []
        pt_vecs = []
        att_probs = []
        reduce_train_set = {}
        hyp = Hypothesis()

        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(action_token_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.use_general_action_embed:
                y_g_tm1_embed = general_action_token_embed[t]

                y_tm1_embed = torch.cat([y_tm1_embed, y_g_tm1_embed], 1)

            if self.dropout_i:
                y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            (h_t, cell_t), att_t, pt_vec, score_t, att_weight = self.step(x, h_tm1,
                                                                          src_encodings, src_encodings_att_linear,
                                                                          src_sent_masks=src_sent_masks)

            if self.sup_attention:
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_actions):
                        action_t = example.tgt_actions[t]
                        #print ("sentence length:",len(example.src_sent))
                        #print (example.src_sent)
                        #print (src_encodings.size())
                        #print (example.src_sent)
                        cand_src_tokens = AttentionUtil.get_candidate_tokens_to_attend(example.src_sent, action_t)
                        #print (action_t)
                        """
                        print ("==========================")
                        print (action_t)
                        print (cand_src_tokens)
                        print (example.src_sent)
                        print ("==========================")
                        """
                        if cand_src_tokens:
                            #print (cand_src_tokens)
                            att_prob = [torch.log(att_weight[e_id, token_id + 1].unsqueeze(0)) for token_id in cand_src_tokens]
                            if len(att_prob) > 1:
                                att_prob = torch.cat(att_prob).sum().unsqueeze(0)
                            else:
                                #print(att_prob[0].size())
                                att_prob = att_prob[0]
                            #print (att_prob.size())
                            #print (att_prob)
                            att_probs.append(att_prob)

            # print (pt_vec.size())
            if isinstance(action_seq[0][t], ReduceAction):
                reduce_train_set[t] = list(hyp.tgt_ids_stack)
                hyp.tgt_ids_stack = hyp.reduce_action_ids(t, action_seq[0][t].rule.body_length)
            else:
                hyp.tgt_ids_stack.append(t)
            scores.append(score_t)
            pt_vecs.append(pt_vec)
            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # (src_sent_len, batch_size, tgt_vocab_size)
        pt_vecs = torch.stack(pt_vecs).squeeze()
        # print (pt_vecs.size())

        sim_scores = []
        for reduce_action_index in sorted(reduce_train_set.keys())[:-1]:
            tgt_ids = reduce_train_set[reduce_action_index]
            stack_embedding = pt_vecs[torch.LongTensor(tgt_ids)]
            # print (stack_embedding.size())
            if self.use_children_lstm_encode:
                stack_embedding, (last_stack_state, last_stack_cell) = self.children_encode_lstm(
                    stack_embedding.unsqueeze(1))
                stack_embedding = stack_embedding.squeeze(1)
            # print(stack_embedding.size())
            reduce_embedding = pt_vecs[reduce_action_index]
            # print(reduce_embedding.size())
            rule_len = action_seq[0][reduce_action_index].rule.body_length
            stack_len = len(tgt_ids)
            stack_label = nn_utils.build_similarity_binary_label(stack_len, rule_len, self.use_cuda)
            # print (stack_embedding.size())
            # print (reduce_embedding.size())
            # print (torch.mm(stack_embedding,reduce_embedding.unsqueeze(1)))
            sim_scores.append((torch.mm(stack_embedding, reduce_embedding.unsqueeze(1)), stack_label))
            if self.mem_net:
                #print(pt_vecs[0].requires_grad)
                pt_vecs = pt_vecs.clone()
                #print (pt_vecs[0].requires_grad)
                reduce_embedding = reduce_embedding.unsqueeze(0)
                stack_embedding = stack_embedding.unsqueeze(0)
                stack_ctx_vec, stack_att_weight = dot_prod_attention(reduce_embedding, stack_embedding, stack_embedding)
                #print (stack_ctx_vec.size())
                #print (reduce_embedding.size())
                pt_vecs[reduce_action_index] = pt_vecs[reduce_action_index] + stack_ctx_vec.squeeze()
        scores = torch.stack(scores)
        #print (att_probs)
        if len(att_probs) > 0:
            att_probs = torch.stack(att_probs).sum()
        else:
            att_probs = None
        #print (att_probs.size())
        if self.sup_attention:
            return scores, sim_scores, att_probs
        else:
            return scores, sim_scores

    def score_decoding_results(self, log_scores, sim_scores, action_seq_var):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        # (tgt_sent_len, batch_size, tgt_vocab_size)
        action_seq_var_sos_omitted = action_seq_var[
                                     1:]  # remove leading <s> in tgt sent, which is not used as the target

        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            tgt_sent_log_scores = -self.label_smoothing_layer(log_scores, action_seq_var_sos_omitted)
        else:
            # (tgt_sent_len, batch_size)
            tgt_sent_log_scores = torch.gather(log_scores, -1, action_seq_var_sos_omitted.unsqueeze(-1)).squeeze(-1)

        tgt_sent_log_scores = tgt_sent_log_scores * (1. - torch.eq(action_seq_var_sos_omitted, 0).float())  # 0 is pad

        # print (tgt_sent_log_scores.size())

        # (batch_size)
        tgt_sent_log_scores = tgt_sent_log_scores.sum(dim=0)

        sim_loss = 0
        for dot_product_score, label in sim_scores:
            # print (dot_product_score.size())
            # print (label.size())
            sim_loss += self.bce_loss(dot_product_score.squeeze(), label.squeeze())

        # print (tgt_sent_log_scores.size())
        loss_s = -tgt_sent_log_scores.unsqueeze(0)[0]
        return sim_loss + loss_s

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = dot_prod_attention(h_t, src_encodings, src_encodings_att_linear, mask=src_sent_masks)
        #print (src_sent_masks)
        # vector for action prediction
        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        if self.dropout:
            att_t = self.dropout(att_t)

        # vector for rule body prediction
        pt_vec = torch.tanh(self.prototype_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        if self.dropout:
            pt_vec = self.dropout(pt_vec)

        # (batch_size, tgt_vocab_size)
        score_t = self.readout(att_t)  # E.q. (6)
        log_score = torch.log_softmax(score_t, dim=-1)

        # score for the hierarchy classification
        # (batch_size, general_action_size)
        """
        if self.hierarchy_label == 'base':
            score_t_b = self.readout_b(att_t)
            score_b = torch.softmax(score_t_b, dim=-1)
            # if self.training:
            # print ("training")
            if self.training:
                log_score_b = torch.log(score_b)
            else:
                log_score_b = score_b.clone().fill_(-float('inf'))

                index = torch.argmax(score_b, dim=-1).unsqueeze(-1)
                src = log_score_b.gather(-1, index).fill_(0)
                log_score_b = log_score_b.scatter(-1, index, src)

            batch_size = log_score_b.size(0)
            for id, index_tensor in self.general_action_vocab.action_hierarchy.items():
                index_size = index_tensor.size(0)
                # print (log_score.size())
                # print (index_tensor.size())
                # print (log_score_b[:,id].expand(index_size, batch_size).T.size())
                log_score = log_score.index_add(-1, index_tensor, log_score_b[:, id].expand(index_size, batch_size).T)
        """
        return (h_t, cell_t), att_t, pt_vec, log_score, alpha_t

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
        # action_seq_var = batch.action_seq_var

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)
        loss = []
        loss_sup = []
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda)
            action_seq_var = tgt_batch.action_seq_var
            if self.sup_attention:
                action_token_logits, sim_score, att_prob = self.decode(src_encodings[:, i, :].unsqueeze(1),
                                                             src_sent_masks[i].unsqueeze(0), (
                                                                 dec_init_vec[0][i, :].unsqueeze(0),
                                                                 dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
                if att_prob:
                #print (att_prob)
                    loss_sup.append(att_prob)
            else:
                action_token_logits, sim_score = self.decode(src_encodings[:, i, :].unsqueeze(1),
                                                             src_sent_masks[i].unsqueeze(0), (
                                                             dec_init_vec[0][i, :].unsqueeze(0),
                                                             dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
            loss_s = self.score_decoding_results(action_token_logits, sim_score, action_seq_var)
            loss.append(loss_s)
        ret_val = [torch.stack(loss)]
        if self.sup_attention:
            ret_val.append(loss_sup)
        # print(loss.data)
        return ret_val

    def infer_rule_body_length(self, embedding_stack, reduce_embedding):
        # print (reduce_embedding.size())
        if self.use_children_lstm_encode:
            embedding_stack, (last_stack_state, last_stack_cell) = self.children_encode_lstm(
                embedding_stack.unsqueeze(1))
            embedding_stack = embedding_stack.squeeze(1)
        length = 0
        score_tensor = torch.sigmoid(torch.mm(embedding_stack, reduce_embedding))
        predicted_vals = score_tensor > 0.5
        for i in range(predicted_vals.size(0) - 1, -1, -1):
            if predicted_vals[i] == 0:
                break
            else:
                length += predicted_vals[i].item()
        return length

    def beam_search(self, example, decode_max_time_step, beam_size=5, relax_factor=5):
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
        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
        # todo change it back
        # eos_id = self.action_vocab['</s>']
        bos_id = self.action_vocab[GenAction(RuleVertex('<s>'))]
        action_vocab_size = len(self.action_vocab)

        first_hyp = Hypothesis()
        first_hyp.action_id.append(bos_id)
        if self.use_general_action_embed:
            general_bos_id = self.general_action_vocab[GenAction(RuleVertex('<s>'))]
            first_hyp.general_action_id.append(general_bos_id)
        first_hyp.actions.append(GenAction(RuleVertex('<s>')))
        # hypotheses = [[bos_id]]
        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step and len(hypotheses) > 0:

            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                src_encodings_att_linear.size(1),
                                                                                src_encodings_att_linear.size(2))

            y_tm1 = new_long_tensor([hyp.action_id[-1] for hyp in hypotheses])
            y_tm1_embed = self.action_embed(y_tm1)
            if self.use_general_action_embed:
                y_g_tm1 = new_long_tensor([hyp.general_action_id[-1] for hyp in hypotheses])
                y_g_tm1_embed = self.general_action_embed(y_g_tm1)

                y_tm1_embed = torch.cat([y_tm1_embed, y_g_tm1_embed], 1)

            if self.dropout_i:
                y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, proto_type_vec, p_t, att_weight = self.step(x, h_tm1,
                                                                              expanded_src_encodings,
                                                                              expanded_src_encodings_att_linear,
                                                                              src_sent_masks=None)

            # p_t = torch.log_softmax(score_t, dim = -1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            #print(new_hyp_scores)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num + relax_factor)
            prev_hyp_ids = top_new_hyp_pos // action_vocab_size
            word_ids = top_new_hyp_pos % action_vocab_size

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            count = 0
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data.tolist(),
                                                           word_ids.cpu().data.tolist(),
                                                           top_new_hyp_scores.cpu().data.tolist()):

                if count >= live_hyp_num:
                    break
                if word_id == self.action_vocab[GenAction(RuleVertex('<gen_pad>'))] or word_id == self.action_vocab[GenAction(RuleVertex('<s>'))] or word_id == self.action_vocab[ReduceAction(Rule(RuleVertex('<re_pad>')))]:
                    continue
                temp_hyp = hypotheses[prev_hyp_id].copy()

                #print (prev_hyp_id)
                # print (len(temp_hyp.embedding_stack))
                temp_hyp.action_id.append(word_id)
                action_instance = self.action_vocab.id2token[word_id]
                if self.use_general_action_embed:
                    temp_hyp.general_action_id.append(self.general_action_vocab[action_instance])
                temp_hyp.actions.append(action_instance)
                action_embedding = proto_type_vec[prev_hyp_id]
                # print (action_embedding.size())
                if isinstance(action_instance, ReduceAction):
                    if len(temp_hyp.embedding_stack) > 0:
                        if str(action_instance.rule.head) == ROOT:
                            action_instance.rule.body_length = len(temp_hyp.heads_stack)
                        else:
                            action_instance.rule.body_length = self.infer_rule_body_length(
                                torch.stack(temp_hyp.embedding_stack), action_embedding.unsqueeze(1))

                            #if action_instance.rule.body_length not in action_instance.rule.body_length_set:
                                #temp_hyp.is_parsable = False
                    else:
                        temp_hyp.is_parsable = False
                    if temp_hyp.is_parsable:
                        temp_hyp.heads_stack = temp_hyp.reduce_actions(action_instance)
                        temp_hyp.embedding_stack = temp_hyp.reduce_embedding(action_embedding,
                                                                             action_instance.rule.body_length, self.mem_net)
                        if temp_hyp.completed():
                            temp_hyp.action_id = temp_hyp.action_id[1:]
                            temp_hyp.actions = temp_hyp.actions[1:]
                            temp_hyp.score = new_hyp_score
                            completed_hypotheses.append(temp_hyp)
                        else:
                            new_hypotheses.append(temp_hyp)
                            live_hyp_ids.append(prev_hyp_id)
                            new_hyp_scores.append(new_hyp_score)
                            count += 1
                    else:
                        continue
                else:
                    temp_hyp.heads_stack.append(action_instance.vertex)
                    temp_hyp.tgt_ids_stack.append(t)
                    # print (len(temp_hyp.embedding_stack))
                    temp_hyp.embedding_stack.append(action_embedding)
                    new_hypotheses.append(temp_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)
                    count += 1

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

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
    def load(cls, model_path, use_cuda=False, use_general_action_embed=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.use_cuda = use_cuda
        saved_args.use_general_action_embed = use_general_action_embed
        parser = cls(vocab, saved_args)

        parser.load_state_dict(saved_state)

        if use_cuda: parser = parser.cuda()
        parser.eval()

        return parser
