# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch import _VF
import numpy as np
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
    print(uninit_vocab)
    return uninit_vocab


def update_vocab_related_parameters(parser, loaded_vocab):
    parser.vocab = loaded_vocab

    pre_trained_src_vocab = parser.src_vocab
    parser.src_vocab = loaded_vocab.source

    pre_trained_src_embedding = parser.src_embed
    parser.src_embed = nn.Embedding(len(parser.src_vocab), parser.embed_size)
    nn.init.xavier_normal_(parser.src_embed.weight.data)

    if parser.embed_fixed:
        # print (type(model.src_embed.weight.grad))
        parser.src_embed.weight.requires_grad = False

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

    print("update ast embedding")
    unit_vertex_vocab = copy_embedding_to_current_layer(pre_trained_vertex_vocab, parser.vertex_vocab,
                                                        pre_trained_ast_embedding, parser.ast_embed)

    pre_trained_action_vocab = parser.action_vocab
    parser.action_vocab = loaded_vocab.action

    pre_trained_action_embedding = parser.readout_action
    parser.readout_action = nn.Linear(parser.embed_size, len(parser.action_vocab), bias=False)
    nn.init.xavier_normal_(parser.readout_action.weight.data)


    print("update action embedding")
    copy_embedding_to_current_layer(pre_trained_action_vocab, parser.action_vocab, pre_trained_action_embedding,
                                    parser.readout_action)


    parser.action_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.action_vocab),
                                                                  ignore_indices=[0, 1])


    print ("update action tokens")

    parser.predicate_tokens = [None] * len(parser.action_vocab.token2id.keys())


    for action_id, action in parser.action_vocab.id2token.items():
        action_tokens = action.prototype_tokens
        if len(action_tokens) == 0:
            from preprocess_data.utils import get_action_predicate
            predicate_name = get_action_predicate(action)
            assert len(predicate_name) > 0
            parser.predicate_tokens[action_id] = predicate_name
        elif 'hidden' in action.prototype_tokens:
            parser.predicate_tokens[action_id] = ['and']
        else:
            parser.predicate_tokens[action_id] = action_tokens


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

@Registrable.register('irnet')
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
        self.embed_fixed = args.embed_fixed
        if self.embed_fixed:
            # print (type(model.src_embed.weight.grad))
            self.src_embed.weight.requires_grad=False

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

        self.att_transform_linear = nn.Linear(self.hidden_size, self.embed_size, bias=False)

        self.predicate_type_embed = nn.Linear(2, self.embed_size)

        # supervised attention
        self.sup_attention = args.sup_attention

        # dropout layer
        self.dropout = args.dropout
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout)

        self.dropout_i = args.dropout_i
        if self.dropout_i > 0:
            self.dropout_i = nn.Dropout(self.dropout_i)

        # project the input stack and reduce head into the embedding space
        self.input_stack_reduce_linear = nn.Linear(self.decoder_size * 2, self.decoder_size,
                                                   bias=False)


        self.readout_action = nn.Linear(self.embed_size, len(self.action_vocab), bias=False)


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
            self.action_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.action_vocab),
                                                                        ignore_indices=[0, 1])

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

        pad_action = self.vocab.action.id2token[0]
        pad_action.prototype_tokens = ['<pad>']

        start_action = self.vocab.action.id2token[1]
        start_action.prototype_tokens = ['<s>']

        self.predicate_tokens = [None] * len(self.vocab.action.token2id.keys())

        self.actiontype_mask = self.new_tensor(len(self.vocab.action.token2id.keys()))

        overlap_predicate_freq = {}
        for action_id, action in self.vocab.action.id2token.items():
            action_tokens = action.prototype_tokens
            if len(action_tokens) == 0:
                from preprocess_data.utils import get_action_predicate
                predicate_name = get_action_predicate(action)
                assert len(predicate_name) > 0
                self.predicate_tokens[action_id] = predicate_name
            elif 'hidden' in action.prototype_tokens:
                self.predicate_tokens[action_id] = ['and']
            else:
                self.predicate_tokens[action_id] = action_tokens
            """
            predicate_str = " ".join(self.predicate_tokens[action_id])
            if predicate_str in overlap_predicate_freq:
                overlap_predicate_freq[predicate_str].append(action_id)
                #self.predicate_tokens[action_id].append(str(overlap_predicate_freq[predicate_str]))
                #self.action_mask[]
            else:
                overlap_predicate_freq[predicate_str] = []
                overlap_predicate_freq[predicate_str].append(action_id)
            """
        """
        for predicate_str, action_ids in overlap_predicate_freq.items():
            if len(action_ids) > 1:
                for id in action_ids:
                    self.actiontype_mask[id] = 1
        print(self.actiontype_mask)
        self.actiontype_mask = self.actiontype_mask.bool().unsqueeze(1).expand(len(self.vocab.action.token2id.keys()), self.embed_size)
        print(self.actiontype_mask.size())
        """
        print(self.predicate_tokens)

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        # print (src_encoding.size())
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

    def encode(self, src_token_embed, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
            dec_init_state, dec_init_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        # print (src_sents_var.size())


        if self.dropout_i:
            src_token_embed = self.dropout_i(src_token_embed)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len, enforce_sorted=False)
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

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        #table_unk_mask = ~table_unk_mask
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            #print (one_table_embedding.size())
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))
            #print (one_table_embedding.size())
            #print(src_embedding.size())
            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)
            #print(topk_val[0])
            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)

        #print (embedding_differ.size())
        #print (table_unk_mask.size())
        #print (table_embedding.size())
        #print (table_unk_mask)


        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(1).expand(
            embedding_differ.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ).bool(), 0)
        #print (embedding_differ.size())
        if self.use_cuda:
            embedding_differ = embedding_differ.cuda()
        return embedding_differ

    def decode(self, src_encodings, src_sent_masks, dec_init_vec, batch, predicate_embedding):
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
        cov_v = torch.zeros(1, requires_grad=False)
        att_weight = torch.zeros(1, requires_grad=False)


        # print (cov_v.size())
        if self.use_cuda:
            cov_v = cov_v.cuda()
            att_weight = att_weight.cuda()

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
                # e_id = 0
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
                    # e_id += 1

            if len(current_node.variables) > 0:
                #h_tm1_v = h_tm1
                input_v = att_tm1
                for var in current_node.variables:
                    # print (x.size())
                    #hyp.v_hidden_embedding = self.variable_lstm(input_v, hyp.v_hidden_embedding)
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
            # (batch_size, tgt_vocab_size)
            #print(predicate_embedding.size())
            #print (att_t.size())

            att_emb = self.att_transform_linear(att_t)

            score_t = torch.mm(predicate_embedding, att_emb.T)  # E.q. (6)
            #print (torch.softmax(score_t.T, dim=-1).sum())
            # print (re_score_t)
            score = torch.log_softmax(score_t.T, dim=-1)
            #print (score)
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


    def gen_x_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        is_list = False
        if type(q[0][0]) == list:
            is_list = True
        for i, one_q in enumerate(q):
            if not is_list:
                q_val = list(
                    map(lambda x: self.src_embed.weight[self.src_vocab[x]], one_q))
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.src_embed.weight[self.src_vocab[w]])
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))

            val_embs.append(q_val)
            val_len[i] = len(q_val)
        max_len = max(val_len)

        val_emb_array = torch.zeros(B, max_len, self.args.embed_size)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]

        if self.args.use_cuda:
            val_inp = val_emb_array.cuda()

        #print (val_inp.size())
        return val_inp

    def forward(self, examples):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """
        src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda, data_type='lambda')

        src_span_list, src_sents_len, predicate_tokens, predicate_map_tensor = src_batch.src_sents_span(self.predicate_tokens)

        if self.use_cuda:
            predicate_map_tensor = predicate_map_tensor.cuda()
        #print (len(predicate_tokens))
        #print(predicate_map_tensor.size())
        predicate_embedding = self.gen_x_batch([predicate_tokens])

        src_token_embed = self.gen_x_batch(src_span_list)

        permute_src_token_embed = src_token_embed.permute(1,0,2)
        #print(predicate_embedding.size())
        #print (src_token_embed[:, 0, :].unsqueeze(1).size())
        #print(src_span_list)
        #print(predicate_map_tensor.size())
        #src_sents_var = src_batch.src_sents_var
        #print(src_batch.src_sents_len)
        # action_seq_var = batch.action_seq_var

        src_encodings, (last_state, last_cell) = self.encode(permute_src_token_embed, src_sents_len)

        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)

        schema_differ = self.embedding_cosine(src_embedding=src_token_embed, table_embedding=predicate_embedding,
                                              table_unk_mask=src_sent_masks)

        #print (src_token_embed.unsqueeze(1).size())
        #print (schema_differ.unsqueeze(3).size())

        #print (schema_differ[0,15,:])
        schema_ctx = (src_token_embed.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        #print (schema_ctx.size())
        schema_embedding = predicate_embedding + schema_ctx

        predicte_type = self.predicate_type_embed(predicate_map_tensor)

        action_type_embedding = self.readout_action.weight.expand(len(examples), self.readout_action.weight.size(0), self.readout_action.weight.size(1))
        #action_type_embedding.data.masked_fill_(~self.actiontype_mask, 0)
        #print (action_type_embedding[0])
        predicate_embedding = schema_embedding + predicte_type + action_type_embedding
        #print((predicate_embedding[0, 6, :] - predicate_embedding[0, 15, :]).sum())
                              #+ action_type_embedding

        #print(predicate_embedding.size())

        loss = []
        loss_sup = []
        # print (len(src_batch))
        for i in range(len(src_batch)):
            tgt_batch = Batch([examples[i]], self.vocab, use_cuda=self.use_cuda, data_type='lambda')
            if self.sup_attention:
                (action_scores, entity_scores, variable_scores), att_prob = self.decode(
                    src_encodings[:, i, :].unsqueeze(1),
                    src_sent_masks[i].unsqueeze(0), (
                        dec_init_vec[0][i, :].unsqueeze(0),
                        dec_init_vec[1][i, :].unsqueeze(0)), tgt_batch, predicate_embedding[i,:,:])
                if att_prob:
                    loss_sup.append(att_prob)
            else:
                (action_scores, entity_scores, variable_scores) = self.decode(src_encodings[:, i, :].unsqueeze(1),
                                                                              src_sent_masks[i].unsqueeze(0), (
                                                                                  dec_init_vec[0][i, :].unsqueeze(0),
                                                                                  dec_init_vec[1][i, :].unsqueeze(0)),
                                                                              tgt_batch, predicate_embedding[i,:,:])
            loss_s = self.score_decoding_results((action_scores, entity_scores, variable_scores), tgt_batch)
            loss.append(loss_s)
        stack_loss = torch.stack(loss)

        ret_val = [stack_loss]
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
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode='token')


        src_span_list, src_sents_len, predicate_tokens, predicate_map_tensor = nn_utils.to_src_sents_span([example.src_sent], self.predicate_tokens)

        if self.use_cuda:
            predicate_map_tensor = predicate_map_tensor.cuda()
        #print (len(predicate_tokens))
        #print(predicate_map_tensor.size())
        predicate_embedding = self.gen_x_batch([predicate_tokens])

        src_token_embed = self.gen_x_batch(src_span_list)

        permute_src_token_embed = src_token_embed.permute(1,0,2)
        #print(predicate_embedding.size())
        #print (src_token_embed[:, 0, :].unsqueeze(1).size())
        #print(src_span_list)
        #print(predicate_map_tensor.size())
        #src_sents_var = src_batch.src_sents_var
        #print(src_batch.src_sents_len)
        # action_seq_var = batch.action_seq_var

        src_encodings, (last_state, last_cell) = self.encode(permute_src_token_embed, src_sents_len)

        #dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)

        schema_differ = self.embedding_cosine(src_embedding=src_token_embed, table_embedding=predicate_embedding,
                                              table_unk_mask=src_sent_masks)


        schema_ctx = (src_token_embed.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

        action_type_embedding = self.readout_action.weight.expand(1, self.readout_action.weight.size(0), self.readout_action.weight.size(1))

        schema_embedding = predicate_embedding + schema_ctx

        predicte_type = self.predicate_type_embed(predicate_map_tensor)

        #action_type_embedding.data.masked_fill_(~self.actiontype_mask, 0)
        #print(action_type_embedding[0])

        predicate_embedding = schema_embedding + predicte_type + action_type_embedding

        #print(predicate_embedding.size())

        # TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        #src_encodings, (last_state, last_cell) = self.encode(src_sents_var, [src_sents_var.size(0)])
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
        vocab_size = len(self.action_vocab)

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
                            (h_t, cell_t), att_t, att_weight = self.step(input_v, hyp.v_hidden_embedding,
                                      src_encodings,
                                      src_encodings_att_linear,
                                      src_sent_masks=None,
                                      lstm_type='variable')
                            score_v_t = self.readout_variable(att_t)
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

                x = torch.stack(x_list)
                h_tm1 = (torch.stack(state_list), torch.stack(cell_list))

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear)


            att_emb = self.att_transform_linear(att_t)

            score_t = torch.mm(predicate_embedding[0,:,:], att_emb.T)  # E.q. (6)
            #print (torch.softmax(score_t.T, dim=-1).sum())
            # print (re_score_t)
            p_t = torch.log_softmax(score_t.T, dim=-1)


                #self.readout_action(att_t)  # E.q. (6)

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
