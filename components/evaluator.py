from __future__ import print_function

import sys, traceback
import numpy as np
from grammar.utils import is_var, is_predicate, is_lit
from common.registerable import Registrable
from grammar.consts import SLOT_PREFIX, VAR_NAME, ROOT, IMPLICIT_HEAD
from grammar.hypothesis import Hypothesis
from grammar.vertex import RuleVertex
from common.utils import config_logger
import os

from preprocess_data.utils import generate_dir

@Registrable.register('default_evaluator')
class DefaultEvaluator(object):
    def __init__(self, args=None):
        self.args = args
        self.default_metric = 'accuracy'
        self.correct_num = 'correct_num'

    def is_hyp_correct(self, example, hyp):
        assert isinstance(hyp, Hypothesis), "hyp should be Hypothesis"
        if self.args.parser == 'seq2seq':
            print(hyp.to_logic_form)
            print(" ".join(example.tgt_code))
            return hyp.to_logic_form == " ".join(example.tgt_code)
        else:
            if self.args.lang.endswith('lambda'):
                return hyp.to_lambda_template == example.to_logic_form
            elif self.args.lang.endswith('prolog'):
                return hyp.to_prolog_template == example.to_logic_form


    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        correct_array = []
        oracle_array = []
        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]
            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    is_correct = self.is_hyp_correct(example, hyp)

                    hyp.is_correct = is_correct

                correct_array.append(hyp_list[0].is_correct)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
            else:
                correct_array.append(False)
                oracle_array.append(False)

        acc = np.average(correct_array)

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc, correct_num = np.sum(correct_array))
        return eval_results