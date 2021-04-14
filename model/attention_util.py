# coding=utf-8
from grammar.action import GenAction, ReduceAction
import torch
from grammar.consts import SLOT_PREFIX, VAR_NAME
from torch import nn


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
        tokens_to_attend = dict()
        if isinstance(action, GenAction):
            tgt_tokens = [inner_token for token in str(action.vertex.rep()).split(' ') for inner_token in token.split('_')]
        elif isinstance(action, ReduceAction):
            tgt_tokens = [inner_token for token in str(action.rule.head.rep()).split(' ') for inner_token in token.split('_')]
        for src_idx, src_token in enumerate(src_tokens):
            src_token = src_token.lower()
            # match lemma
            for tgt_token in tgt_tokens:
                tgt_token = tgt_token.lower()
                if tgt_token == SLOT_PREFIX.lower() or tgt_token == VAR_NAME.lower():
                    continue

                if len(src_token) >= 2 and (tgt_token.startswith(src_token) or src_token.startswith(tgt_token) or\
                                src_token in LOGICAL_FORM_LEXICON.get(tgt_token, [])):
                    tokens_to_attend[src_idx] = src_token
        return tokens_to_attend
