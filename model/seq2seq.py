# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from components.dataset import Batch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from common.registerable import Registrable
from model import nn_utils
from grammar.rule import ReduceAction
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
import os


@Registrable.register('seq2seq')
class Seq2SeqModel(nn.Module):
    """
    a standard seq2seq model
    """
    def __init__(self, vocab, args):
        super(Seq2SeqModel, self).__init__()
        self.use_cuda = args.use_cuda
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.vocab = vocab
        self.args = args
        self.src_vocab = vocab.source
        self.tgt_vocab = vocab.code

        self.src_embed = nn.Embedding(len(self.src_vocab), self.embed_size)

        self.tgt_emb = nn.Embedding(len(self.tgt_vocab), self.embed_size)

        self.encoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)
        self.decoder_lstm = nn.LSTMCell(self.embed_size + self.hidden_size, self.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(self.hidden_size, len(self.tgt_vocab), bias=False)

        # dropout layer
        self.dropout = args.dropout
        self.dropout = nn.Dropout(self.dropout)
        self.decoder_word_dropout = args.decoder_word_dropout
        self.label_smoothing = None

        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.tgt_vocab), ignore_indices=[0])


    def encode(self, src_sents_var, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
            dec_init_state, dec_init_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        #print (src_sents_var.size())
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
        dec_init_cell = self.decoder_cell_init(enc_last_cell)
        #print (dec_init_cell.squeeze().size())
        dec_init_state = torch.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    def decode(self, src_encodings, src_sents_len, dec_init_vec, tgt_sents_var):
        """
        compute the final softmax layer at each decoding step
        :param src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
        :param src_sents_len: list[int]
        :param dec_init_vec: tuple((batch_size, hidden_size))
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            scores: Variable(src_sent_len, batch_size, src_vocab_size)
        """
        new_tensor = src_encodings.data.new
        batch_size = src_encodings.size(1)

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        #print (src_encodings.size())
        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # initialize the attentional vector
        att_tm1 = new_tensor(batch_size, self.hidden_size).zero_()
        assert (att_tm1.requires_grad == False, "the att_tm1 requires grad is False")
        # (batch_size, src_sent_len)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)
        #print (src_sent_masks)
        #print (src_sent_masks.size())
        # (tgt_sent_len, batch_size, embed_size)
        tgt_token_embed = self.tgt_emb(tgt_sents_var)
        #print (tgt_token_embed.size())
        scores = []
        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(tgt_token_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim
            y_tm1_embed = y_tm1_embed.squeeze(0)

            x = torch.cat([y_tm1_embed, att_tm1], 1)
            #print (x.size())
            #print (src_sent_masks.size())
            (h_t, cell_t), att_t, score_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)

            scores.append(score_t)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # (src_sent_len, batch_size, tgt_vocab_size)
        scores = torch.stack(scores)

        return scores

    def score_decoding_results(self, log_scores, tgt_sents_var):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        tgt_sents_var_sos_omitted = tgt_sents_var[1:]  # remove leading <s> in tgt sent, which is not used as the target

        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            tgt_sent_log_scores = -self.label_smoothing_layer(log_scores, tgt_sents_var_sos_omitted)
        else:
            # (tgt_sent_len, batch_size)
            tgt_sent_log_scores = torch.gather(log_scores, -1, tgt_sents_var_sos_omitted.unsqueeze(-1)).squeeze(-1)

        tgt_sent_log_scores = tgt_sent_log_scores * (1. - torch.eq(tgt_sents_var_sos_omitted, 0).float())  # 0 is pad
        # (batch_size)
        tgt_sent_log_scores = tgt_sent_log_scores.sum(dim=0)

        return tgt_sent_log_scores.unsqueeze(0)

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        #print (h_tm1[1].size())
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = Seq2SeqModel.dot_prod_attention(h_t,
                                                         src_encodings, src_encodings_att_linear,
                                                         mask=src_sent_masks)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        # (batch_size, tgt_vocab_size)
        score_t = self.readout(att_t)  # E.q. (6)
        log_score = torch.log_softmax(score_t, dim=-1)

        return (h_t, cell_t), att_t, log_score

    def forward(self, examples):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """
        batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        src_sents_var = batch.src_sents_var
        src_sents_len = batch.src_sents_len
        tgt_seq_var = batch.tgt_seq_var

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        tgt_token_logits = self.decode(src_encodings, src_sents_len, dec_init_vec, tgt_seq_var)
        tgt_sent_log_scores = self.score_decoding_results(tgt_token_logits, tgt_seq_var)
        loss = -tgt_sent_log_scores[0]
        #print (tgt_sent_log_scores[0])
        # print(loss.data)
        return [loss]

    @staticmethod
    def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
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
        #print (att_weight)
        #print (torch.sum(att_weight[0]))
        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def greedy_search(self, example, decode_max_time_step):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode ='token')


        #TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
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

        att_tm1 = torch.zeros(1, self.hidden_size, requires_grad=False)
        hyp_scores = torch.zeros(1, requires_grad=False)
        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
        # todo change it back
        eos_id = self.tgt_vocab['</s>']
        bos_id = self.tgt_vocab['<s>']
        tgt_vocab_size = len(self.tgt_vocab)

        first_hyp = Hypothesis()
        first_hyp.tgt_code_tokens_id.append(bos_id)
        first_hyp.tgt_code_tokens.append('<s>')



        t = 0
        while t < decode_max_time_step:
            t += 1

            y_tm1 = new_long_tensor([first_hyp.tgt_code_tokens_id[-1]])
            y_tm1_embed = self.tgt_emb(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, p_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=None)

            new_hyp_score = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            #print(new_hyp_scores.size())
            top_new_hyp_scores, top_new_hyp_pos = p_t.max(1)

            word_id = top_new_hyp_pos

            first_hyp.tgt_code_tokens_id.append(word_id.item())
            tgt_token = self.tgt_vocab.id2token[word_id.item()]

            first_hyp.tgt_code_tokens.append(tgt_token)
            if word_id == eos_id:
                first_hyp.tgt_code_tokens_id = first_hyp.tgt_code_tokens_id[1:-1]
                first_hyp.tgt_code_tokens = first_hyp.tgt_code_tokens[1:-1]
                first_hyp.score = new_hyp_score
                break


            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t


        return [first_hyp]

    def beam_search(self, example, decode_max_time_step, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode ='token')

        #TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
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

        att_tm1 = torch.zeros(1, self.hidden_size, requires_grad=False)
        hyp_scores = torch.zeros(1, requires_grad=False)
        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
        # todo change it back
        eos_id = self.tgt_vocab['</s>']
        bos_id = self.tgt_vocab['<s>']
        tgt_vocab_size = len(self.tgt_vocab)

        first_hyp = Hypothesis()
        first_hyp.tgt_code_tokens_id.append(bos_id)
        first_hyp.tgt_code_tokens.append('<s>')
        #hypotheses = [[bos_id]]
        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))
            """
            for hyp in hypotheses:
                print ([str(action) for action in hyp.actions])
            """
            y_tm1 = new_long_tensor([hyp.tgt_code_tokens_id[-1] for hyp in hypotheses])
            y_tm1_embed = self.tgt_emb(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, p_t = self.step(x, h_tm1,
                                                      expanded_src_encodings, expanded_src_encodings_att_linear,
                                                      src_sent_masks=None)

            #p_t = torch.log_softmax(score_t, dim = -1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            #print(new_hyp_scores.size())
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos // tgt_vocab_size

            word_ids = top_new_hyp_pos % tgt_vocab_size

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data.tolist(), word_ids.cpu().data.tolist(), top_new_hyp_scores.cpu().data.tolist()):

                temp_hyp = hypotheses[prev_hyp_id].copy()
                temp_hyp.tgt_code_tokens_id.append(word_id)
                tgt_token = self.tgt_vocab.id2token[word_id]

                temp_hyp.tgt_code_tokens.append(tgt_token)
                if word_id == eos_id:
                    temp_hyp.tgt_code_tokens_id = temp_hyp.tgt_code_tokens_id[1:-1]
                    temp_hyp.tgt_code_tokens = temp_hyp.tgt_code_tokens[1:-1]
                    temp_hyp.score = new_hyp_score
                    completed_hypotheses.append(temp_hyp)
                else:
                    new_hypotheses.append(temp_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = new_float_tensor(new_hyp_scores)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
        if len(completed_hypotheses) == 0:
            dummy_hyp = Hypothesis()
            completed_hypotheses.append(dummy_hyp)
        else:
            #completed_hypotheses = [hyp for hyp in completed_hypotheses if hyp.completed()]
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
    def load(cls, model_path, use_cuda=False, args = None):
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
