# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch._six import inf
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from grammar.utils import is_lit, is_var
from grammar.action import *
from grammar.vertex import *
import math
from torch.nn.parameter import Parameter
from grammar.rule import *
from six.moves import xrange

from model.rat_sql.models.spider.spider_match_utils import compute_schema_linking


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class BatchLinear(nn.Module):
    def __init__(self, batch_size, label_len, hidden_size, mask):
        super(BatchLinear, self).__init__()
        self.batch_size = batch_size
        self.label_len = label_len
        self.hidden_size = hidden_size
        self.weight = Parameter(data=torch.Tensor(batch_size, label_len, hidden_size), requires_grad = True)
        self.mask = mask

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        """
        :param input: (batch_size, hidden_size)
        """
        #print (input.size())
        #print (self.weight.size())
        batch_score = torch.bmm(self.weight, input.unsqueeze(2)).squeeze(2)
        if self.mask is not None:
            batch_score = batch_score.flatten()[self.mask.flatten()]

        return batch_score

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        #print(vector)
        vector = vector * mask  + (mask + 1e-45).log()
        #print (vector)
        #print (torch.nn.functional.softmax(vector, dim=dim))
    return torch.nn.functional.log_softmax(vector, dim=dim)

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

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


def length_array_to_mask_tensor(length_array, use_cuda=False, valid_entry_has_mask_one=False):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.zeros((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        if valid_entry_has_mask_one:
            mask[i][:seq_len] = 1
        else:
            mask[i][seq_len:] = 1

    mask = torch.tensor(mask, dtype=torch.bool)
    return mask.cuda() if use_cuda else mask

def build_similarity_binary_label(length_stack, length_rule, use_cuda=False):
    stack_label = torch.zeros(length_stack)
    stack_label[-length_rule:] = 1
    if use_cuda:
        stack_label = stack_label.cuda()
    return stack_label

def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def token2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2token(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2token[w] for w in s] for s in sents]
    else:
        return [vocab.id2token[w] for w in sents]


def to_seq_pad(sequences, append_boundary_sym=True, mode ='token'):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        if mode == 'token':
            sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]
            pad = '<pad>'
        elif mode == 'action':
            bos = GenAction(RuleVertex('<s>'))
            sequences = [[bos] + seq for seq in sequences]
            pad = GenAction(RuleVertex('<gen_pad>'))
        elif mode == 're_action':
            pad = ReduceAction(Rule(RuleVertex('<re_pad>')))
        elif mode == 'ast':
            bos = RuleVertex('<s>')
            sequences = [[bos] + seq for seq in sequences]
            pad = RuleVertex('<gen_pad>')
        else:
            raise ValueError

    max_len = max(len(s) for s in sequences)
    batch_size = len(sequences)

    sents_t = []
    for i in range(batch_size):
        sents_t.append([])
        for k in range(max_len):
            if k < len(sequences[i]):
                sents_t[i].append(sequences[i][k])
            else:
                sents_t[i].append(pad)

    return sents_t

def to_src_sents_span(src_sents, predicate_tokens):
    #sequences = [['<s>'] + seq + ['</s>'] for seq in src_sents]



    # [action.prototype_tokens for action in self.vocab.action.token2id.keys()]
    #print (sequences[0])
    #print (predicate_tokens)
    #print (action_vocab.token2id)
    batch_size = len(src_sents)
    predicate_map_tensor = torch.zeros(batch_size, len(predicate_tokens), 2)

    src_span_list = []
    for src_id, src_scent in enumerate(src_sents):
        link_dict = compute_schema_linking(src_scent, predicate_tokens)
        seq = ['<s>'] + src_scent + ['</s>']

        question_map_set = dict()
        #predicate_map_set = dict()
        for link_id, link_type in link_dict['q_col_match'].items():
            link_ids = link_id.split(',')
            question_id = int(link_ids[0]) + 1

            predicate_id = int(link_ids[1])
            question_map_set[question_id] = predicate_id
            #question_map_set.add(question_id)
            #predicate_map_set[predicate_id] = link_type
            if link_type == 'CEM':
                predicate_map_tensor[src_id, predicate_id, 0] = 1
            elif link_type == 'CPM':
                predicate_map_tensor[src_id, predicate_id, 1] = 1
        src_span = []

        sub_span = ['predicatetype']
        for token_idx, token in enumerate(seq):
            if token_idx in question_map_set:
                if token_idx - 1 in question_map_set and not (question_map_set[token_idx - 1] == question_map_set[token_idx]):
                    src_span.append(sub_span)
                    sub_span = ['predicatetype']
                    sub_span.append(token)
                else:
                    sub_span.append(token)
            else:
                if len(sub_span) > 1:
                    src_span.append(sub_span)
                    sub_span = ['predicatetype']
                src_span.append([token])
        if len(sub_span) > 1:
            src_span.append(sub_span)
        src_span_list.append(src_span)

    src_lengths = [len(src_sent_span) for src_sent_span in src_span_list]

    return src_span_list, src_lengths, predicate_tokens, predicate_map_tensor


def to_input_variable(sequences, vocab, use_cuda=False, append_boundary_sym=True, mode ='token'):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        if mode == 'token':
            sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]
            pad = '<pad>'
        elif mode == 'action':
            bos = GenAction(RuleVertex('<s>'))
            sequences = [[bos] + seq for seq in sequences]
            pad = GenAction(RuleVertex('<gen_pad>'))
        elif mode == 're_action':
            pad = ReduceAction(Rule(RuleVertex('<re_pad>')))
        elif mode == 'ast':
            bos = RuleVertex('<s>')
            sequences = [[bos] + seq for seq in sequences]
            pad = RuleVertex('<gen_pad>')
        elif mode == 'lit':
            pad = '<pad>'
        else:
            raise ValueError
    else:
        pad = '<pad>'
    #print (sequences)
    token_ids = token2id(sequences, vocab)
    sents_t = input_transpose(token_ids, vocab[pad])
    #print (sents_t)
    sents_var = torch.LongTensor(sents_t)
    #print (sents_var.size())
    if use_cuda:
        sents_var = sents_var.cuda()

    return sents_var


def to_hierarchy_input_variable(sequences, vocab_dict, use_cuda=False, append_boundary_sym=True):
    if append_boundary_sym:
        sequences = [[GenAction(RuleVertex('<s>'))] + seq for seq in sequences]
    #print (sequences)
    hierarchy_var = {}
    hierarchy_seq = {}
    for type_name, vocab in vocab_dict.items():
        hierarchy_seq[type_name] = []
    for seq in sequences:
        for type_name, vocab in vocab_dict.items():
            hierarchy_seq[type_name].append([])
        for action in seq:
            hierarchy_seq[type(action).__name__][-1].append(action)
    #print(hierarchy_seq)
    for type_name, seqs in hierarchy_seq.items():
        token_ids = token2id(seqs, vocab_dict[type_name])
        pad_action = vocab_dict[type_name].id2token[0]
        sents_t = input_transpose(token_ids, vocab_dict[type_name][pad_action])
        sents_var = torch.LongTensor(sents_t)
        if use_cuda:
            sents_var = sents_var.cuda()
        hierarchy_var[type_name] = sents_var
    return hierarchy_var


def variable_constr(x, v, use_cuda=False):
    return Variable(torch.cuda.x(v)) if use_cuda else Variable(torch.x(v))


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in xrange(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
       source: https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def direct_clip_grad_norm_(grads, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(grads, torch.Tensor):
        grads = [grads]
    #print (grads[0].data)
    grads = list(filter(lambda g: g is not None, grads))
    #print (len(grads))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in grads)
    else:
        total_norm = 0
        for g in grads:
            param_norm = g.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return total_norm


def uniform_init(lower, upper, params):
    for p in params:
        p.data.uniform_(lower, upper)


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal_(p.data)


def identity(x):
    return x


class LabelSmoothing(nn.Module):
    """Implement label smoothing.

    Reference: the annotated transformer
    """

    def __init__(self, smoothing, tgt_vocab_size, ignore_indices=None, use_cuda = True):
        if ignore_indices is None: ignore_indices = []

        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        smoothing_value = smoothing / float(tgt_vocab_size - 1 - len(ignore_indices))
        self.one_hot = torch.zeros((tgt_vocab_size,)).fill_(smoothing_value)
        if use_cuda:
            self.one_hot = self.one_hot.cuda()
        for idx in ignore_indices:
            self.one_hot[idx] = 0.

        self.confidence = 1.0 - smoothing
        self.one_hot = self.one_hot.unsqueeze(0)
        #self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, model_prob, target):
        # (batch_size, *, tgt_vocab_size)
        dim = list(model_prob.size())[:-1] + [1]
        #print (self.one_hot)
        true_dist = Variable(self.one_hot, requires_grad=False).repeat(*dim)
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        # true_dist = model_prob.data.clone()
        # true_dist.fill_(self.smoothing / (model_prob.size(1) - 1))  # FIXME: no label smoothing for <pad> <s> and </s>
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(model_prob, true_dist).sum(dim=-1)


class FeedForward(nn.Module):
    """Feed forward neural network adapted from AllenNLP"""

    def __init__(self, input_dim, num_layers, hidden_dims, activations, dropout):
        super(FeedForward, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore

        self.activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))

        self.linear_layers = nn.ModuleList(linear_layers)
        dropout_layers = [nn.Dropout(p=value) for value in dropout]
        self.dropout = nn.ModuleList(dropout_layers)
        self.output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def forward(self, x):
        output = x
        for layer, activation, dropout in zip(self.linear_layers, self.activations, self.dropout):
            output = dropout(activation(layer(output)))
        return output
