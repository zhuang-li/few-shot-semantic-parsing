# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
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
from model.locked_dropout import LockedDropout
from model.attention_util import AttentionUtil, Attention
import os

from model.nn_utils import masked_log_softmax


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
    for current_idx, current_token in current_vocab.id2token.items():
        if current_token in is_exist_vocab:
            pre_trained_id = pre_trained_vocab[current_token]
            # print (pre_trained_layer.weight.data[pre_trained_id].size())
            # print (pre_trained_layer.weight.data.size())
            current_layer.weight.data[current_idx].copy_(pre_trained_layer.weight.data[pre_trained_id])
            if current_mask is not None:
                # print (current_mask.size())
                # print (pre_trained_mask.size())
                current_mask[0][current_idx] = pre_trained_mask[0][pre_trained_id]
        else:
            uninit_vocab[current_token] = current_idx
    print(uninit_vocab)
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

    parser.train_action_set.add(GenAction(RuleVertex('<gen_pad>')))
    parser.train_action_set.add(GenAction(RuleVertex('<s>')))
    """
    node_action_set = set()
    for action in parser.train_action_set:
        if isinstance(action, ReduceAction):
            node_action_set.add(action.rule.head.copy())
        elif isinstance(action, GenAction):
            node_action_set.add(action.vertex.copy())
    print (node_action_set)
    """
    print("update ast embedding")
    unit_vertex_vocab = copy_embedding_to_current_layer(pre_trained_vertex_vocab, parser.vertex_vocab,
                                                        pre_trained_ast_embedding, parser.ast_embed)

    # ==================================================================================================================

    pre_trained_re_vocab = parser.re_vocab
    parser.re_vocab = loaded_vocab.re_action

    pre_trained_re_embedding = parser.re_embedding
    parser.re_embedding = nn.Embedding(len(parser.re_vocab), parser.hidden_size)

    nn.init.xavier_normal_(parser.re_embedding.weight.data)

    parser.re_action_proto = parser.new_tensor(len(parser.re_vocab), parser.hidden_size)

    pre_trained_re_action_mask = parser.re_action_mask
    parser.re_action_mask = torch.zeros(1, len(parser.re_vocab))

    print("update action embedding")
    copy_embedding_to_current_layer(pre_trained_re_vocab, parser.re_vocab, pre_trained_re_embedding,
                                    parser.re_embedding, pre_trained_re_action_mask, parser.re_action_mask,
                                    parser.train_action_set)

    print(parser.re_action_mask)

    parser.re_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.re_vocab),
                                                                  ignore_indices=[])

    if parser.use_cuda:
        parser.re_action_mask = parser.re_action_mask.cuda()

    # ==================================================================================================================

    pre_trained_gen_vocab = parser.gen_vocab
    parser.gen_vocab = loaded_vocab.gen_action

    pre_trained_gen_embedding = parser.gen_embedding
    parser.gen_embedding = nn.Embedding(len(parser.gen_vocab), parser.hidden_size)

    nn.init.xavier_normal_(parser.gen_embedding.weight.data)

    parser.gen_action_proto = parser.new_tensor(len(parser.gen_vocab), parser.hidden_size)

    pre_trained_gen_action_mask = parser.gen_action_mask
    parser.gen_action_mask = torch.zeros(1, len(parser.gen_vocab))

    print("update action embedding")
    copy_embedding_to_current_layer(pre_trained_gen_vocab, parser.gen_vocab, pre_trained_gen_embedding,
                                    parser.gen_embedding, pre_trained_gen_action_mask, parser.gen_action_mask,
                                    parser.train_action_set)

    print(parser.gen_action_mask)

    parser.gen_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.gen_vocab),
                                                              ignore_indices=[0,1])

    if parser.use_cuda:
        parser.gen_action_mask = parser.gen_action_mask.cuda()

    # ==================================================================================================================

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




@Registrable.register('seq2seq_few_shot_bin')
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

        self.att_reg = args.att_reg
        if self.att_reg:
            # self.att_reg_cond_linear = nn.Linear(1, 1, bias=False)
            # self.att_reg_strsim_linear = nn.Linear(1, 1, bias=True)
            # todo change it back
            # self.l1_criterion = torch.nn.L1Loss(size_average=False)
            self.att_reg_linear = nn.Linear(self.hidden_size, 1)
        # project the input stack and reduce head into the embedding space
        self.input_stack_reduce_linear = nn.Linear(self.decoder_size * 2, self.decoder_size,
                                                   bias=False)

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

        self.gen_vocab = vocab.gen_action
        self.re_vocab = vocab.re_action


        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.gen_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.gen_vocab),
                                                                     ignore_indices=[0, 1])

            self.re_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.re_vocab),
                                                                    ignore_indices=[0])

            self.entity_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.entity_vocab),
                                                                        ignore_indices=[0])
            if len(self.variable_vocab) > 2:
                self.variable_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing,
                                                                              len(self.variable_vocab),
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

        # action proto
        # prediction layer of the gen actions
        self.gen_embedding = nn.Embedding(len(self.gen_vocab), self.hidden_size)
        nn.init.xavier_normal_(self.gen_embedding.weight.data)

        self.gen_action_proto = self.new_tensor(len(self.gen_vocab), self.hidden_size)

        self.gen_action_mask = torch.zeros(1, len(self.gen_vocab))

        self.gen_proto_mask = None

        if args.use_cuda:
            self.gen_action_mask = self.gen_action_mask.cuda()

        # prediction layer of the reduce actions

        self.re_embedding = nn.Embedding(len(self.re_vocab), self.hidden_size)
        nn.init.xavier_normal_(self.re_embedding.weight.data)

        self.re_action_proto = self.new_tensor(len(self.re_vocab), self.hidden_size)

        self.re_action_mask = torch.zeros(1, len(self.re_vocab))

        self.re_proto_mask = None

        if args.use_cuda:
            self.re_action_mask = self.re_action_mask.cuda()


        self.train_action_set = set()
        self.train_mask = args.train_mask

    def init_action_embedding(self, examples, few_shot_mode='proto_train'):

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
        # print (len(src_batch))
        att_logits_list = []
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda)
            # print ("================")
            # print (src_encodings[:, i, :].unsqueeze(1).size())
            res_dict = self.decode(
                src_encodings[:, i, :].unsqueeze(1),
                src_sent_masks[i].unsqueeze(0), (
                    dec_init_vec[0][i, :].unsqueeze(0),
                    dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
            # print ("================")
            # print (res_dict['att_logits'].size())
            att_logits_list.append(res_dict['att_logits'])

        # new_action_proto.requires_grad = True
        # print (new_action_proto.requires_grad)
        # print(new_action_proto[11].requires_grad)
        # print (self.action_proto[10])
        new_re_action_proto = self.new_tensor(self.re_action_proto.size()).zero_()
        re_action_count = {}

        new_gen_action_proto = self.new_tensor(self.gen_action_proto.size()).zero_()
        gen_action_count = {}

        action_set = set()



        for template_id, template_seq in enumerate(src_batch.action_seq):

            att_seq = att_logits_list[template_id]
            # print (att_seq.size())
            # print (len(template_seq))
            for att_id, action in enumerate(template_seq):

                if isinstance(action, ReduceAction):
                    self.assign_proto_embedding(new_re_action_proto, re_action_count, action_set, action, self.re_vocab,
                                           few_shot_mode, att_seq[att_id])
                elif isinstance(action, GenAction):
                    self.assign_proto_embedding(new_gen_action_proto, gen_action_count, action_set, action, self.gen_vocab,
                                           few_shot_mode, att_seq[att_id])


        self.init_proto_embedding(re_action_count, new_re_action_proto, self.re_embedding, self.re_action_mask, few_shot_mode)
        self.init_proto_embedding(gen_action_count, new_gen_action_proto, self.gen_embedding, self.gen_action_mask, few_shot_mode)
            # print(self.action_embedding.weight[10])

        return action_set

    def init_proto_embedding(self, action_count, new_action_proto, action_embedding, action_mask, few_shot_mode):
        for action_vocab_id, freq in action_count.items():
            new_action_proto[action_vocab_id] = new_action_proto[action_vocab_id] / freq

            assert (new_action_proto[action_vocab_id].requires_grad == True)

        if few_shot_mode == 'proto_train':
            self.action_proto = new_action_proto
            # make_dot(self.action_proto[5])
        elif few_shot_mode == 'fine_tune':
            # proto_embedding = self.action_proto.detach()
            # print ((self.action_embedding.weight.data.t()*label_mask).t())
            # print (new_action_proto)
            # print (self.action_mask)
            # print (new_action_proto)
            action_embedding.weight.data.copy_(
                new_action_proto + (action_embedding.weight.data.t() * action_mask).t())


    def assign_proto_embedding(self, new_action_proto, action_count, action_set, action, action_vocab, few_shot_mode, att):
        action_id = action_vocab[action]

        if few_shot_mode == 'proto_train':
            # if action_id == 3:
            #   print (new_action_proto[action_id])
            action_set.add(action)
            new_action_proto[action_id] = new_action_proto[action_id] + att
            # if action_id == 3:
            #   print (new_action_proto[action_id])
            # print(new_action_proto[action_id].requires_grad)
            # print (action)
            if action_id in action_count:
                action_count[action_id] += 1
            else:
                action_count[action_id] = 1
        elif few_shot_mode == 'fine_tune':
            if not (action in self.train_action_set):
                # print (self.train_action_set)
                # print ("asdasd")
                # print (att_seq[att_id])
                # print (action)
                new_action_proto[action_id] = new_action_proto[action_id] + att
                # print (new_action_proto[action_id])
                if action_id in action_count:
                    action_count[action_id] += 1
                else:
                    action_count[action_id] = 1



    def get_action_score(self, action_embedding, action_proto, action_mask, action_vocab, att_t):
        if self.few_shot_mode == 'proto_train':
            action_proto = action_proto
        elif self.few_shot_mode == 'supervised_train':
            action_proto = action_embedding(self.new_long_tensor([i for i in range(len(action_vocab))]))
            # print (self.action_mask)
            action_proto = (action_proto.t() * action_mask).t()
            # print (action_proto)
        elif self.few_shot_mode == 'fine_tune':
            action_proto = action_embedding(self.new_long_tensor([i for i in range(len(action_vocab))]))
            # print (action_proto.size())
            # print (action_proto.size())
        # expaned_action_proto = action_proto.expand(batch_size, len(self.action_vocab), self.hidden_size)
        if self.metric == 'prototype':
            score_t = -nn_utils.euclidean_dist(action_proto, att_t)  # E.q. (6)
        elif self.metric == 'dot':
            # action_proto : vocab_length * hidden_size ||| att_t : batch_size * hidden_size = vocab_length *  1
            # print (action_proto.size())
            # print (att_t.size())
            score_t = torch.mm(action_proto, att_t.t())  # E.q. (6)
        # score_t = self.readout_action(torch.cat([expaned_action_proto, expanded_att], dim=-1)).squeeze(-1)
        score_t = score_t.t()
        if self.train_mask:
            if self.few_shot_mode == 'supervised_train':
                score = masked_log_softmax(score_t, action_mask)
            elif self.few_shot_mode == 'proto_train':
                # print (self.proto_mask)
                score = masked_log_softmax(score_t, action_mask)
                # print (self.proto_mask)
                # print(torch.softmax(score_t, dim=-1))
            elif self.few_shot_mode == 'fine_tune':
                score = torch.log_softmax(score_t, dim=-1)
        else:
            score = torch.log_softmax(score_t, dim=-1)

        return score

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
        src_seq_len = src_encodings.size(1)
        # print (src_encodings.size())
        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # print (src_encodings.size())
        # initialize the attentional vector
        att_tm1 = new_tensor(batch_size, self.hidden_size).zero_()
        assert att_tm1.requires_grad == False, "the att_tm1 requires grad is False"
        # (batch_size, src_sent_len)

        # print (src_sent_masks.size())
        # (tgt_sent_len, batch_size, embed_size)
        ast_seq_embed = self.ast_embed(ast_seq_var)
        if len(batch.variable_seq[0]) > 0:
            var_seq_embed = self.variable_embed(batch.variable_seq_var)

        # if len(batch.entity_seq[0]) > 0:
        # entity_seq_embed = self.entity_embed(batch.entity_seq_var)

        gen_scores = []
        re_scores = []

        variable_scores = []
        entity_scores = []
        att_probs = []

        entity_att_probs = []

        reg_loss_list = []
        hyp = Hypothesis()
        hyp.hidden_embedding_stack.append(h_tm1)
        hyp.v_hidden_embedding = h_tm1
        cov_v = torch.zeros(1, requires_grad=False)
        att_weight = torch.zeros(1, requires_grad=False)
        # print (cov_v.size())
        if self.use_cuda:
            cov_v = cov_v.cuda()
            att_weight = att_weight.cuda()

        att_logits = []
        v_id = 0
        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(ast_seq_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim

            if self.use_coverage:
                # print (cov_v.unsqueeze(-1).size())
                # print (src_encodings_att_linear.clone().size())
                cov_v = cov_v + att_weight
                # print (torch.sigmoid(self.fertility(h_tm1[0][0])))
                src_encodings_att_linear_cov = src_encodings_att_linear.clone() + self.cov_linear(
                    torch.div(cov_v.unsqueeze(-1), torch.sigmoid(self.fertility(h_tm1[0][0]))))
            else:
                src_encodings_att_linear_cov = src_encodings_att_linear

            y_tm1_embed = y_tm1_embed.squeeze(0)

            # action instance
            current_node = action_seq[0][t]

            # if self.dropout_i:
            # y_tm1_embed = self.dropout_i(y_tm1_embed.unsqueeze(0)).squeeze(0)

            if self.use_att:
                x = torch.cat([y_tm1_embed, att_tm1], 1)
            else:
                x = y_tm1_embed

            if isinstance(current_node, ReduceAction):
                # non-terminal
                body_length = current_node.rule.body_length

                # print (torch.stack(input_stack[:-body_length]).squeeze(1).mean(dim=0).unsqueeze(0).size())
                if self.use_input_lstm_encode:
                    input_lstm_embedding, (last_input_stack_state, last_input_stack_cell) = self.input_encode_lstm(
                        torch.stack(hyp.heads_embedding_stack[-body_length:]))
                    children_embedding = last_input_stack_state.squeeze(0)
                    # print(last_input_stack_state.size())
                else:
                    children_embedding = torch.stack(hyp.heads_embedding_stack[-body_length:]).squeeze(1).mean(
                        dim=0).unsqueeze(0)
                    # print (children_embedding.size())
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
                    # print (x.size())
                    # h_tm1_e = self.entity_lstm(x, h_tm1_e)

                    (h_e, cell_e), att_t_e, att_weight_e = self.step(x, h_tm1_e, src_encodings,
                                                                     src_encodings_att_linear_cov,
                                                                     src_sent_masks=src_sent_masks, lstm_type='entity')

                    score_e_t = self.readout_entity(att_t_e)
                    score_e = torch.log_softmax(score_e_t, dim=-1)
                    entity_scores.append(score_e)
                    h_tm1_e = (h_e, cell_e)
                    # print (att_weight_e.size())

                    entity_align_id = current_node.entity_align[e_id]
                    if not (entity_align_id == -1):
                        # print (current_node)
                        # print (batch.src_sents)
                        # print (entity_align_id)
                        # print (att_weight_e[0,entity_align_id])
                        # print (att_weight_e[0,entity_align_id])
                        # print (att_weight_e[0][entity_align_id])
                        entity_att_probs.append(torch.log(att_weight_e[0, entity_align_id]))

                    e_id += 1

            if len(current_node.variables) > 0:
                # h_tm1_v = h_tm1
                input_v = att_tm1
                for var in current_node.variables:
                    # print (x.size())
                    # hyp.v_hidden_embedding = self.variable_lstm(input_v, hyp.v_hidden_embedding)
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
            # if self.dropout_i:
            # x = self.dropout_i(x.unsqueeze(0)).squeeze(0)

            (h_t, cell_t), att_t, att_weight = self.step(x_point, h_tm1_point, src_encodings,
                                                         src_encodings_att_linear_cov,
                                                         src_sent_masks=src_sent_masks)

            predict_action_instance = action_seq[0][t + 1]
            if isinstance(predict_action_instance, ReduceAction):
                #print (self.re_action_mask)
                re_score = self.get_action_score(self.re_embedding, self.re_action_proto, self.re_action_mask, self.re_vocab, att_t)
                re_scores.append(re_score)
            elif isinstance(predict_action_instance, GenAction):
                #print (self.gen_action_mask)
                gen_score = self.get_action_score(self.gen_embedding, self.gen_action_proto, self.gen_action_mask,
                                                 self.gen_vocab, att_t)
                gen_scores.append(gen_score)


            if self.att_reg and (not (
                    len(predict_action_instance.prototype_tokens) == 1 and (predict_action_instance.prototype_tokens[
                0] == SLOT_PREFIX or predict_action_instance.prototype_tokens[0] == VAR_NAME or
                    predict_action_instance.prototype_tokens[0] == 'e'))):
                #print ("sdadsada")
                pad_list = [0] * (src_seq_len - len(predict_action_instance.string_sim_score))
                cond_tensor = self.new_tensor(predict_action_instance.cond_score + pad_list).unsqueeze(0)

                strsim_tensor = self.new_tensor(predict_action_instance.string_sim_score + pad_list).unsqueeze(0)
                # print (self.att_reg_cond_linear(cond_tensor).size())
                # reg_weight = self.att_reg_cond_linear(cond_tensor).t() + (torch.nn.functional.relu(self.att_reg_strsim_linear(strsim_tensor))).t()
                # print (cond_tensor.t())

                reg_weight = torch.nn.functional.sigmoid(self.att_reg_linear(att_t)) * cond_tensor + (
                        (1 - torch.nn.functional.sigmoid(self.att_reg_linear(att_t))) * strsim_tensor)
                # print (reg_weight)
                reg_weight = torch.div(reg_weight, reg_weight.sum())
                # print (reg_weight)

                # print (action_seq[0][t + 1])
                # print (batch.examples[0].src_sent)

                # print (reg_weight)
                # print (att_weight.size())
                # print (action_seq[0][t + 1])
                # print (reg_weight.size())
                # print (batch.examples[0].src_sent)
                # print (batch.src_sents_var.size())
                # print (src_encodings.size())
                # reg_loss = self.l1_criterion(att_weight, reg_weight)
                # print (self.att_reg_cond_linear(cond_tensor).t())
                reg_loss = torch.abs(att_weight - reg_weight)
                reg_loss = reg_loss.sum()
                # print (reg_loss)
                # print (reg_loss)
                reg_loss_list.append(reg_loss)

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
            att_logits.append(att_t)
            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        gen_scores = torch.stack(gen_scores)
        re_scores = torch.stack(re_scores)

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

        res_dict['scores'] = (gen_scores, re_scores, entity_scores, variable_scores)
        res_dict['att_logits'] = att_logits

        if self.sup_attention:
            res_dict['att_probs'] = att_probs

        if self.att_reg:
            # print (torch.stack(reg_loss_list).size())
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

        gen_scores,re_scores, entity_scores, variable_scores = scores
        action_var = batch.action_seq_var[1:]

        gen_action_var = batch.hierarchy_action_seq_var[GenAction.__name__]
        re_action_var = batch.hierarchy_action_seq_var[ReduceAction.__name__]
        #print (gen_action_var.size())
        #print (re_action_var.size())
        if entity_scores is not None:
            entity_var = batch.entity_seq_var
        if variable_scores is not None:
            variable_var = batch.variable_seq_var
        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            # print (nn_scores.size())
            #print (gen_action_var.size())
            #print (re_scores.size())
            #print (re_action_var.size())
            # print (action_scores.size())
            # print (action_var.size())
            gen_sent_log_scores = -self.gen_label_smoothing_layer(gen_scores, gen_action_var)
            re_sent_log_scores = -self.re_label_smoothing_layer(re_scores, re_action_var)

            if entity_scores is not None:
                entity_log_scores = -self.entity_label_smoothing_layer(entity_scores, entity_var)
            if variable_scores is not None:
                if len(self.variable_vocab) > 2:
                    variable_log_scores = -self.variable_label_smoothing_layer(variable_scores, variable_var)
                else:
                    variable_log_scores = torch.gather(variable_scores, -1, variable_var.unsqueeze(-1)).squeeze(-1)

        else:
            # (tgt_sent_len, batch_size)
            gen_sent_log_scores = torch.gather(gen_scores, -1, gen_action_var.unsqueeze(-1)).squeeze(-1)
            re_sent_log_scores = torch.gather(re_scores, -1, re_action_var.unsqueeze(-1)).squeeze(-1)

            if entity_scores is not None:
                entity_log_scores = torch.gather(entity_scores, -1, entity_var.unsqueeze(-1)).squeeze(-1)
            if variable_scores is not None:
                variable_log_scores = torch.gather(variable_scores, -1, variable_var.unsqueeze(-1)).squeeze(-1)

        gen_sent_log_scores = gen_sent_log_scores * (1. - torch.eq(gen_action_var, 0).float())  # 0 is pad

        re_sent_log_scores = re_sent_log_scores * (1. - torch.eq(re_action_var, 0).float())

        gen_sent_log_scores = gen_sent_log_scores.sum(dim=0)
        # print (gen_sent_log_scores)
        # print (gen_sent_log_scores)
        # (batch_size)
        re_sent_log_scores = re_sent_log_scores.sum(dim=0)
        # print (gen_sent_log_scores)

        # generate action loss
        gen_loss = -gen_sent_log_scores.unsqueeze(0)[0]
        # reduce action loss
        re_loss = -re_sent_log_scores.unsqueeze(0)[0]

        if entity_scores is not None:
            entity_log_scores = entity_log_scores * (1. - torch.eq(entity_var, 0).float())  # 0 is pad

            entity_log_scores = entity_log_scores.sum(dim=0)

            entity_loss = -entity_log_scores.unsqueeze(0)[0]
        if variable_scores is not None:
            variable_log_scores = variable_log_scores * (1. - torch.eq(variable_var, 0).float())  # 0 is pad

            variable_log_scores = variable_log_scores.sum(dim=0)

            variable_loss = -variable_log_scores.unsqueeze(0)[0]
        loss = gen_loss + re_loss
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
        # print (src_sent_masks)
        # vector for action prediction
        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        if self.dropout:
            att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t

    def get_bi_action_score(self, action_proto, action_embedding, action_vocab, action_mask, att_t):
        if self.few_shot_mode == 'proto_train':
            action_proto = action_proto
        elif self.few_shot_mode == 'supervised_train':
            action_proto = action_embedding(self.new_long_tensor([i for i in range(len(action_vocab))]))
            action_proto = (action_proto.t() * action_mask).t()
        elif self.few_shot_mode == 'fine_tune':
            action_proto = action_embedding(self.new_long_tensor([i for i in range(len(action_vocab))]))
            # print (action_proto)
        # expanded_att = unsqueezed_att.expand(unsqueezed_att.size(0), len(self.action_vocab), self.hidden_size)
        # expaned_action_proto = action_proto.expand(unsqueezed_att.size(0), len(self.action_vocab), self.hidden_size)
        # score_t = self.readout_action(torch.cat([expaned_action_proto, expanded_att], dim=-1)).squeeze(-1)
        # print (score_t.size())
        if self.metric == 'prototype':
            score_t = -nn_utils.euclidean_dist(action_proto, att_t)  # E.q. (6)
        elif self.metric == 'dot':
            # action_proto : vocab_length * hidden_size ||| att_t : batch_size * hidden_size = vocab_length *  1
            score_t = torch.mm(action_proto, att_t.t())
        # hyp.current_gen_emb = current_att_t
        # print (score_t.size())
        score_t = score_t.t()

        # hyp.current_gen_emb = current_att_t

        p_t = torch.log_softmax(score_t, dim=-1)

        return p_t

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
        loss_reg = []
        loss_ent = []
        # print (len(src_batch))
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda)
            # print ("================")
            # print (src_encodings[:, i, :].unsqueeze(1).size())
            res_dict = self.decode(
                src_encodings[:, i, :].unsqueeze(1),
                src_sent_masks[i].unsqueeze(0), (
                    dec_init_vec[0][i, :].unsqueeze(0),
                    dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch)
            if self.sup_attention and res_dict['att_probs']:
                # print (res_dict['att_probs'])
                loss_sup.append(res_dict['att_probs'])

            loss_s = self.score_decoding_results(res_dict['scores'], tgt_batch)

            if self.att_reg:
                # print (res_dict['reg_loss'])
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
        # print(loss.data)
        return ret_val

    def assign_nodes_back(self, node, padding, origin_node_list, c=0):

        for child in node.children:
            # print (hyp.heads_stack[-1])
            # print (child)
            if child.has_children():
                c = self.assign_nodes_back(child, padding, origin_node_list, c)
            else:
                if isinstance(child, RuleVertex):
                    if child.head.startswith(padding):
                        child.head = origin_node_list[c]
                        c += 1
                else:
                    if child.vertex.head.startswith(padding):
                        child.head = origin_node_list[c]
                        c += 1
        return c

    def beam_search(self, example, decode_max_time_step, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        # print ("============================")
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode='token')

        entity_align = []

        for id, token in enumerate(example.src_sent):
            if is_lit(token, self.args.lang):
                entity_align.append(id)

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

        vocab_size = re_vocab_size + gen_vocab_size

        #assert (re_vocab_size + gen_vocab_size == vocab_size), "re vocab + gen vocab == vocab_size"

        first_hyp = Hypothesis()
        first_hyp.tgt_ids_stack.append(bos_id)
        # first_hyp.embedding_stack.append(att_tm1)
        first_hyp.hidden_embedding_stack.append(h_tm1)
        first_hyp.current_att_cov = torch.zeros(1, src_sents_var.size(0), requires_grad=False)
        first_hyp.var_id.append(1)
        first_hyp.ent_id.append(1)
        first_hyp.v_hidden_embedding = h_tm1
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
                                                                                    src_encodings_att_linear.size(
                                                                                        2)).clone()

                for id, hyp in enumerate(hypotheses):
                    hyp.current_att_cov = hyp.current_att_cov + att_w_tm1[id].view(hyp.current_att_cov.size())
                    # print (t)
                    # print (hyp.current_att_cov)
                    expanded_src_encodings_att_linear[id] = expanded_src_encodings_att_linear[id] + self.cov_linear(
                        torch.div(hyp.current_att_cov.unsqueeze(-1), torch.sigmoid(self.fertility(h_tm1[id][0]))))

            else:
                expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                    src_encodings_att_linear.size(1),
                                                                                    src_encodings_att_linear.size(2))

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
                x_list = []
                state_list = []
                cell_list = []
                state, cell = h_tm1
                for id, hyp in enumerate(hypotheses):
                    # print (hyp.actions[-1])
                    if hyp.heads_stack[-1].is_terminal():
                        hyp.heads_embedding_stack.append(x[id])
                        x_list.append(x[id])
                        state_list.append(state[id])
                        cell_list.append(cell[id])

                    else:
                        # non-terminal
                        body_length = hyp.actions[-1].rule.body_length
                        # print (torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(dim=0))
                        # print (torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(dim=0).unsqueeze(0).size())
                        # print (x[id].size())

                        if self.use_input_lstm_encode:
                            # print (torch.stack(hyp.heads_embedding_stack[-body_length:]).size())
                            input_lstm_embedding, (
                                last_input_stack_state, last_input_stack_cell) = self.input_encode_lstm(
                                torch.stack(hyp.heads_embedding_stack[-body_length:]).unsqueeze(1))
                            children_embedding = last_input_stack_state.squeeze(0)
                            # print (last_input_stack_state.size())
                        else:
                            children_embedding = torch.stack(hyp.heads_embedding_stack[-body_length:]).mean(
                                dim=0).unsqueeze(0)
                            # print (children_embedding.size())
                        x_point = self.input_stack_reduce_linear(torch.cat(
                            [children_embedding,
                             x[id].unsqueeze(0)], 1)).squeeze()
                        hyp.heads_embedding_stack = hyp.heads_embedding_stack[:-body_length]
                        # assert not torch.isnan(x_point).any(), "x has no nan"
                        # print (x_point.size())
                        x_list.append(x_point)
                        state_point, cell_point = hyp.hidden_embedding_stack[-(body_length + 1)]
                        state_list.append(state_point)
                        cell_list.append(cell_point)
                        hyp.hidden_embedding_stack = hyp.hidden_embedding_stack[:-body_length]
                        hyp.heads_embedding_stack.append(x_point)


                    if len(hyp.actions[-1].entities) > 0:
                        h_tm1_e = (h_tm1[0][id].unsqueeze(0), h_tm1[1][id].unsqueeze(0))
                        ent_list = []
                        # argument_id = 0
                        for ent in hyp.actions[-1].entities:
                            # print (len(x_list))
                            # print (x_list[id].size())
                            # print (h_tm1_v[0].size())
                            # h_t_e, cell_t_e = self.entity_lstm(x[id].unsqueeze(0), h_tm1_e)

                            (h_t_e, cell_t_e), att_t_e, att_weight_e = self.step(x[id].unsqueeze(0), h_tm1_e,
                                                                                 src_encodings,
                                                                                 src_encodings_att_linear,
                                                                                 src_sent_masks=None,
                                                                                 lstm_type='entity')

                            score_e_t = self.readout_entity(att_t_e)
                            score_e = torch.log_softmax(score_e_t, dim=-1)
                            # print (score_v)

                            score_e[0][0] = -1000.0
                            # score_e[0][1] = -1000.0
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

                        assert c == len(hyp.actions[
                                            -1].entities), "variables {} and nodes variables {} num must be in consistency {} and {}".format(
                            c, len(hyp.actions[-1].entities), hyp.heads_stack[-1], hyp.actions[-1])

                    if len(hyp.actions[-1].variables) > 0:
                        # h_tm1_v = (h_tm1[0][id].unsqueeze(0), h_tm1[1][id].unsqueeze(0))
                        var_list = []
                        input_v = att_tm1[id].unsqueeze(0)
                        for var in hyp.actions[-1].variables:
                            # print (len(x_list))
                            # print (x_list[id].size())
                            # print (h_tm1_v[0].size())
                            # h_t, cell_t = self.variable_lstm(input_v,
                            #                                hyp.v_hidden_embedding)
                            (h_t, cell_t), att_t, att_weight = self.step(input_v, hyp.v_hidden_embedding,
                                                                         src_encodings,
                                                                         src_encodings_att_linear,
                                                                         src_sent_masks=None,
                                                                         lstm_type='variable')
                            score_v_t = self.readout_variable(att_t)

                            score_v = torch.log_softmax(score_v_t, dim=-1)
                            # print (score_v)
                            score_v[0][0] = -1000.0
                            # score_v[0][1] = -1000.0
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
                            # print (child)
                        assert c == len(hyp.actions[
                                            -1].variables), "variables {} and nodes variables {} num must be in consistency {} and {}".format(
                            c, len(hyp.actions[-1].variables), hyp.heads_stack[-1], hyp.actions[-1])

                x = torch.stack(x_list)
                h_tm1 = (torch.stack(state_list), torch.stack(cell_list))

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear,
                                                         src_sent_masks=None)

            re_p_t = self.get_bi_action_score(self.re_action_proto, self.re_embedding, self.re_vocab, self.re_action_mask, att_t)

            gen_p_t = self.get_bi_action_score(self.gen_action_proto, self.gen_embedding, self.gen_vocab,
                                              self.gen_action_mask, att_t)
            #print (re_p_t.size())
            #re_p_t[0][0] = -1000.0
            #gen_p_t[0][0] = -1000.0
            #gen_p_t[0][1] = -1000.0
            #print (torch.exp(re_p_t))
            #print (torch.exp(gen_p_t))
            p_t = torch.cat([re_p_t, gen_p_t], dim = -1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            # print(new_hyp_scores)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            #print ()
            prev_hyp_ids = top_new_hyp_pos // vocab_size
            word_ids = top_new_hyp_pos % vocab_size
            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data.tolist(),
                                                           word_ids.cpu().data.tolist(),
                                                           top_new_hyp_scores.cpu().data.tolist()):
                temp_hyp = hypotheses[prev_hyp_id].copy()


                # print (action_instance)
                if word_id < re_vocab_size:
                    action_instance = self.re_vocab.id2token[word_id]
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
                    word_id = word_id - re_vocab_size
                    temp_hyp.reduce_action_count = 0
                    if word_id == self.gen_vocab[GenAction(RuleVertex('<gen_pad>'))] or word_id == self.gen_vocab[
                        GenAction(RuleVertex('<s>'))]:
                        # print ("continue")
                        continue

                    action_instance = self.gen_vocab.id2token[word_id]

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
            hyp_scores = new_float_tensor(new_hyp_scores)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
            t += 1
        if len(completed_hypotheses) == 0:

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
            'vocab': self.vocab,

            're_action': self.re_action_proto,
            're_action_mask': self.re_action_mask,

            'gen_action': self.gen_action_proto,
            'gen_action_mask': self.gen_action_mask,

            'train_action': self.train_action_set,
        }

        torch.save(params, path)

    @classmethod
    def load(cls, model_path, use_cuda=False, loaded_vocab=None):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']

        saved_args.use_cuda = use_cuda
        parser = cls(vocab, saved_args)
        parser.load_state_dict(saved_state)
        parser.train_action_set = params['train_action']

        parser.re_action_proto.copy_(params['re_action'])
        parser.re_action_mask.copy_(params['re_action_mask'])

        parser.gen_action_proto.copy_(params['gen_action'])
        parser.gen_action_mask.copy_(params['gen_action_mask'])

        src_vocab = None
        vertex_vocab = None
        if loaded_vocab:
            src_vocab, vertex_vocab = update_vocab_related_parameters(parser, loaded_vocab)

        if use_cuda:
            parser = parser.cuda()

        return parser, src_vocab, vertex_vocab
