# coding=utf-8
from grammar.action import GenAction, ReduceAction
import torch
from grammar.consts import SLOT_PREFIX, VAR_NAME
from torch import nn

"""

"""

LOGICAL_FORM_LEXICON = {
    'city:t': ['citi'],
    'density:i': ['densiti', 'averag', 'popul'],
    'loc:t' : ['where','in'],
    'next_to:t' : ['border', 'neighbor', 'surround'],
    'argmax' : ['highest', 'largest', 'most', 'greatest', 'longest', 'biggest', "high"],
    'argmin' : ['shortest', 'smallest', 'least', 'lowest'],
    'count' : ['mani', 'number'],
    'sum' : ['total'],
    'size:i' : ['big','biggest','largest'],
    'population:i' : ['peopl', 'popul', 'citizen'],
    '>' : ['-er', 'than'],
    '=': ['equal'],
    '\+': ['not'],
    'req': ['require', 'degree'],
    'deg': ['degree'],
    'exp': ['experience'],
    'salary' : ['salari']
}


class AttentionUtil(object):
    @staticmethod
    def get_candidate_tokens_to_attend(src_tokens, action):
        #print(src_tokens)
        tokens_to_attend = dict()
        if isinstance(action, GenAction):
            tgt_tokens = [inner_token for token in str(action.vertex.rep()).split(' ') for inner_token in token.split('_')]
        elif isinstance(action, ReduceAction):
            tgt_tokens = [inner_token for token in str(action.rule.head.rep()).split(' ') for inner_token in token.split('_')]
        #print (action)
        for src_idx, src_token in enumerate(src_tokens):
            src_token = src_token.lower()
            # match lemma
            for tgt_token in tgt_tokens:
                tgt_token = tgt_token.lower()
                if tgt_token == SLOT_PREFIX.lower() or tgt_token == VAR_NAME.lower():
                    continue
                #print (tgt_token)
                if len(src_token) >= 2 and (tgt_token.startswith(src_token) or src_token.startswith(tgt_token) or\
                                src_token in LOGICAL_FORM_LEXICON.get(tgt_token, [])):
                    tokens_to_attend[src_idx] = src_token
        #print ("=============")
        #print (src_token)
        #print (tgt_tokens)
        #print(tokens_to_attend)
        #print ("=============")
        return tokens_to_attend


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    :return:
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = torch.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight



class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, batch_size, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            print ("dot attention !")
            pass
        elif method == 'general':
            print ("general attention !")
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            print ("concat attention !")
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            print ("bahdanau attention !")
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError


    def forward(self, last_hidden, encoder_outputs, mask=None):
        batch_size, seq_lens, _ = encoder_outputs.size()

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)
        # attn_energies = Variable(torch.zeros(batch_size, seq_lens))  # B x S

        if mask is not None:
            attention_energies = attention_energies.data.masked_fill_(mask, -float('inf'))

        return torch.softmax(attention_energies, dim=-1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`)
        :return:
        """

        # assert last_hidden.size() == torch.Size([batch_size, self.hidden_size]), last_hidden.size()
        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = torch.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = torch.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)
