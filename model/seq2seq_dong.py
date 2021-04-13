# coding=utf-8
import numpy as np





import torch
import torch.nn as nn
import torch.nn.utils
from components.dataset import Batch
from common.registerable import Registrable
from model import nn_utils
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
import os

@Registrable.register('seq2seq_dong')
class Seq2SeqModel(nn.Module):
    """
    a standard seq2seq model
    """
    def __init__(self, vocab, args):
        super(Seq2SeqModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.src_vocab = vocab.source
        self.tgt_vocab = vocab.code
        self.encoder = RNN(args, len(self.src_vocab))
        self.decoder = RNN(args, len(self.tgt_vocab))
        self.decode_max_time_step = args.decode_max_time_step
        self.attention_decoder = AttnUnit(args, len(vocab.code))
        self.criterion = nn.NLLLoss(size_average=False, ignore_index=0)

        # settings
        self.use_cuda = args.use_cuda
        self.hidden_size = args.hidden_size
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attention_decoder = self.attention_decoder.cuda()



    def forward(self, batch_examples):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)

        """
        examples = [e for e in batch_examples if len(e.tgt_code) <= self.decode_max_time_step]
        batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        src_sents_var = batch.src_sents_var
        enc_batch = torch.transpose(src_sents_var, 0, 1)
        src_sents_len = batch.src_sents_len
        #print (enc_batch)
        #print (src_sents_len)

        enc_max_len = max(src_sents_len)
        tgt_seq_var = batch.tgt_seq_var
        dec_batch = torch.transpose(tgt_seq_var, 0, 1)
        tgt_code_len = batch.tgt_code_len
        dec_max_len = max(tgt_code_len) - 1
        batch_size = enc_batch.size(0)
        enc_outputs = torch.zeros((batch_size, enc_max_len, self.encoder.hidden_size), requires_grad=True)
        if self.use_cuda:
            enc_outputs = enc_outputs.cuda()

        enc_s = {}
        for j in range(enc_max_len + 1):
            enc_s[j] = {}

        dec_s = {}

        for j in range(self.decode_max_time_step + 1):
            dec_s[j] = {}

        for i in range(1, 3):
            enc_s[0][i] = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)
            dec_s[0][i] = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)
            if self.use_cuda:
                enc_s[0][i] = enc_s[0][i].cuda()
                dec_s[0][i] = dec_s[0][i].cuda()

        for i in range(enc_max_len):
            enc_s[i + 1][1], enc_s[i + 1][2] = self.encoder(enc_batch[:, i], enc_s[i][1], enc_s[i][2])
            enc_outputs[:, i, :] = enc_s[i + 1][2]

        loss = 0

        for i in range(batch_size):

            dec_s[0][1][i, :] = enc_s[src_sents_len[i]][1][i, :]
            dec_s[0][2][i, :] = enc_s[src_sents_len[i]][2][i, :]

        for i in range(dec_max_len):
            dec_s[i + 1][1], dec_s[i + 1][2] = self.decoder(dec_batch[:, i], dec_s[i][1], dec_s[i][2])
            pred = self.attention_decoder(enc_outputs, dec_s[i + 1][2])
            loss += self.criterion(pred, dec_batch[:, i + 1])

        return [loss]

    def greedy_search(self, example):
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode ='token')
        src_sents_var = torch.transpose(src_sents_var, 0, 1)
        end = src_sents_var.size(1)

        prev_c = torch.zeros((1, self.encoder.hidden_size), requires_grad=False)
        prev_h = torch.zeros((1, self.encoder.hidden_size), requires_grad=False)
        enc_outputs = torch.zeros((1, end, self.encoder.hidden_size), requires_grad=False)
        if self.use_cuda:
            prev_c = prev_c.cuda()
            prev_h = prev_h.cuda()
            enc_outputs = enc_outputs.cuda()
        # TODO check that c,h are zero on each iteration

        # reversed order
        for i in range(end):
            # TODO verify that this matches the copy_table etc in sample.lua
            cur_input = src_sents_var[0][i]
            if self.use_cuda:
                cur_input = cur_input.cuda()
            prev_c, prev_h = self.encoder(cur_input, prev_c, prev_h)
            enc_outputs[:, i, :] = prev_h
        # encoder_outputs = torch.stack(encoder_outputs).view(-1, end, encoder.hidden_size)
        eos_id = self.tgt_vocab['</s>']
        bos_id = self.tgt_vocab['<s>']
        # decode
        first_hyp = Hypothesis()

        text_gen = []
        if self.use_cuda >= 0:
            prev_word = torch.tensor([bos_id], dtype=torch.long).cuda()
        else:
            prev_word = torch.tensor([bos_id], dtype=torch.long)
        while True:
            prev_c, prev_h = self.decoder(prev_word, prev_c, prev_h)
            pred = self.attention_decoder(enc_outputs, prev_h)

            _, _prev_word = pred.max(1)
            prev_word = _prev_word.resize(1)
            if (prev_word[0] == eos_id) or (
                    len(text_gen) >= self.decode_max_time_step):
                break
            else:
                text_gen.append(prev_word[0])
        first_hyp.tgt_code_tokens_id = text_gen
        tgt_list = []
        for idx in text_gen:
            tgt_list.append(self.tgt_vocab.id2token[idx.item()])
        first_hyp.tgt_code_tokens = tgt_list
        return [first_hyp]

    def beam_search(self, example, decode_max_time_step, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode ='token')
        """
        src_list = []
        for idx in src_sents_var.tolist():
            src_list.append(self.src_vocab.id2token[idx[0]])
        """
        src_sents_var = torch.transpose(src_sents_var, 0, 1)
        end = src_sents_var.size(1)

        prev_c = torch.zeros((1, self.encoder.hidden_size), requires_grad=False)
        prev_h = torch.zeros((1, self.encoder.hidden_size), requires_grad=False)
        enc_outputs = torch.zeros((1, end, self.encoder.hidden_size), requires_grad=False)
        if self.use_cuda:
            prev_c = prev_c.cuda()
            prev_h = prev_h.cuda()
            enc_outputs = enc_outputs.cuda()
        # TODO check that c,h are zero on each iteration
        # reversed order
        for i in range(end):
            # TODO verify that this matches the copy_table etc in sample.lua
            cur_input = src_sents_var[0][i]
            if self.use_cuda:
                cur_input = cur_input.cuda()
            prev_c, prev_h = self.encoder(cur_input, prev_c, prev_h)
            enc_outputs[:, i, :] = prev_h

        new_long_tensor = torch.cuda.LongTensor
        new_float_tensor = torch.cuda.FloatTensor

        eos_id = self.tgt_vocab['</s>']
        bos_id = self.tgt_vocab['<s>']

        if self.use_cuda >= 0:
            hyp_scores = torch.zeros(1, requires_grad=False).cuda()
        else:
            hyp_scores = torch.zeros(1, requires_grad=False)

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

            if self.use_cuda >= 0:
                prev_word = torch.tensor([hyp.tgt_code_tokens_id[-1] for hyp in hypotheses], dtype=torch.long).cuda()

            else:
                prev_word = torch.tensor([hyp.tgt_code_tokens_id[-1] for hyp in hypotheses], dtype=torch.long)

            prev_c, prev_h = self.decoder(prev_word, prev_c, prev_h)

            enc_outputs_expand = enc_outputs.expand(prev_h.size(0), enc_outputs.size(1), enc_outputs.size(2))
            p_t = self.attention_decoder(enc_outputs_expand, prev_h)

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
            prev_h,  prev_c= prev_h[live_hyp_ids], prev_c[live_hyp_ids]

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

class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.hidden_size, 4 * opt.hidden_size)
        self.h2h = nn.Linear(opt.hidden_size, 4*opt.hidden_size)

    def forward(self, x, prev_c, prev_h):
        gates = self.i2h(x) \
            + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
        return cy, hy

class RNN(nn.Module):
    def __init__(self, opt, input_size):
        super(RNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.lstm = LSTM(self.opt)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input_src, prev_c, prev_h):
        src_emb = self.embedding(input_src) # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h)
        return prev_cy, prev_hy

class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size

        self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top):

        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0,2,1), attention)
        hid = torch.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2),dec_s_top), 1)))
        h2y_in = hid
        if self.opt.dropout > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)


        return pred