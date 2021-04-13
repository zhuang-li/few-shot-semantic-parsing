# coding=utf-8
from __future__ import print_function

import sys
import traceback
from model import seq2seq
from model import seq2seq_dong
from model import seq2seq_cross_templates
from model import seq2seq_c_t_stack
from model import seq2seq_action
from model import seq2seq_bi_action
from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    for example in examples:
        if args.parser == "seq2seq_c_t":
            hyps = model.beam_search(example, args.decode_max_time_step, beam_size=args.beam_size, relax_factor=args.relax_factor)
        else:
            hyps = model.beam_search(example, args.decode_max_time_step, beam_size=args.beam_size)

        decode_results.append(hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
