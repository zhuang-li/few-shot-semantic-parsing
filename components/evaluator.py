from __future__ import print_function

import sys, traceback
import numpy as np

from components.validate_grammar import read_grammar, validate_logic_form
from continual_model.domain_overnight import OvernightDomain
from grammar.utils import is_var, is_predicate, is_lit
from common.registerable import Registrable
from grammar.consts import SLOT_PREFIX, VAR_NAME, ROOT, IMPLICIT_HEAD
from grammar.hypothesis import Hypothesis
from grammar.vertex import RuleVertex
from model import seq2seq
from model import seq2seq_dong
from model import seq2seq_cross_templates
from model import seq2seq_c_t_stack
from model import seq2seq_action
from model import seq2seq_bi_action
from sklearn.metrics import classification_report
from common.utils import config_logger
import os

from preprocess_data.utils import generate_dir, parse_overnight_query_helper, parse_lambda_query_helper, \
    parse_lambda_query

logger = config_logger("evaluation_result")


def turn_candidate_to_template(candidate, data_type):
    candidate_template_list = []
    for token in candidate.split(' '):
        if is_var(token, dataset=data_type):
            token = VAR_NAME
        elif is_lit(token, dataset=data_type):
            token = SLOT_PREFIX
        candidate_template_list.append(token)

    return (" ".join(candidate_template_list)).strip()


def write_to_file(file_path, candidate_list, reference_list, reference_template_list, src_list, action_list, data_type):
    assert len(candidate_list) == len(reference_list), "the two list must have the same length {} {}".format(
        len(candidate_list), len(reference_list))
    dir_path = os.path.dirname(file_path)
    template_result_file = "template_result.txt"
    complete_result_file = "complete_result.txt"
    template_result_file_path = os.path.join(dir_path, template_result_file)
    generate_dir(dir_path)

    complete_result_file_path = os.path.join(dir_path, complete_result_file)
    generate_dir(dir_path)

    full_result_file = "full_result.txt"
    full_result_file_path = os.path.join(dir_path, full_result_file)

    f = open(template_result_file_path, 'w')
    for id, candidate in enumerate(candidate_list):
        reference_template = reference_template_list[id]
        candidate_template = turn_candidate_to_template(candidate, data_type=data_type)
        f.write(reference_template + '\t' + candidate_template + '\n')
    f.close()

    f = open(complete_result_file_path, 'w')
    for id, candidate in enumerate(candidate_list):
        reference = reference_list[id]

        f.write(reference + '\t' + candidate + '\n')
    f.close()

    f = open(full_result_file_path, 'w')
    for id, candidate in enumerate(candidate_list):
        src_sent = src_list[id]
        f.write(src_sent + '\n')
        reference = reference_list[id]
        f.write(reference + '\n')
        f.write(candidate + '\n')
        actions = action_list[id]
        f.write(actions + '\n')
    f.close()


@Registrable.register('default_evaluator')
class DefaultEvaluator(object):
    def __init__(self, args=None):
        self.args = args
        self.default_metric = 'accuracy'
        self.correct_num = 'correct_num'
        if self.args.validate_grammar:
            self.grammar, self.transition_system = read_grammar(lang=self.args.lang)

    def is_hyp_correct(self, example, hyp):
        assert isinstance(hyp, Hypothesis), "hyp should be Hypothesis"
        # print (self.args.parser)
        if self.args.parser == 'seq2seq' or self.args.parser == 'seq2seq_dong':
            logical_form = hyp.to_logic_form
            if self.args.lang.endswith('prolog'):
                logical_form = '( ' + hyp.to_logic_form + ' )'
            if logical_form == example.to_logic_form:
                return True
            else:
                """
                print("=======================================================")
                print("Source Sentence")
                print(" ".join(example.src_sent))
                print("Token Candidate")
                print(hyp.to_logic_form)
                print("Token Reference")
                print(example.to_logic_form)
                print("=======================================================")
                """
                return False
        elif self.args.parser == 'seq2seq_complete' or \
                self.args.parser == 'seq2seq_few_shot' or \
                self.args.parser == 'seq2seq_fsc' or \
                self.args.parser == 'seq2seq_few_shot_bin' or \
                self.args.parser == 'seq2seq_few_shot_plain' or \
                self.args.parser == 'seq2seq_topdown' or \
                self.args.parser == 'ratsql' or \
                self.args.parser == 'gem' or \
                self.args.parser == 'icarl' or \
                self.args.parser == 'emr' or \
                self.args.parser == 'ea_emr' or \
                self.args.parser == 'origin' or \
                self.args.parser == 'independent' or \
                self.args.parser == 'gem_emr' or \
                self.args.parser == 'loss_emr'or \
                self.args.parser == 'loss_gem' or \
                self.args.parser == 'gss_emr' or \
                self.args.parser == 'a_gem' or \
                self.args.parser == 'adap_emr' or \
                self.args.parser == 'irnet' or \
                self.args.parser == 'emr_wo_t': \
                # print ("sdadadadadsa")
            if self.args.lang.endswith('lambda'):
                if hyp.to_lambda_template == example.to_logic_form:
                    return True
                else:
                    """
                    print ("=======================================================")
                    print ("Source Sentence")
                    print (" ".join(example.src_sent))
                    print ("Token Candidate")
                    print (hyp.to_lambda_template)
                    #print (hyp.actions)
                    print ("Token Reference")
                    print (example.to_logic_form)
                    #print (example.tgt_actions)
                    print ("=======================================================")
                    """
                    return False
            elif self.args.lang.endswith('prolog'):
                if hyp.to_prolog_template == example.to_logic_form:
                    return True
                else:
                    """
                    print ("=======================================================")
                    print ("Source Sentence")
                    print (" ".join(example.src_sent))
                    print ("Token Candidate")
                    print (hyp.to_prolog_template)
                    print ("Token Reference")
                    print (example.to_logic_form)
                    print ("=======================================================")
                    """
                    return False
        else:
            if self.args.lang == "geo_prolog" or self.args.lang == "job_prolog" or self.args.lang == "nlmap":
                if hyp.to_prolog_template == example.to_prolog_template:
                    return True
                else:
                    """
                    print("=======================================================")
                    print("Source Sentence")
                    print(" ".join(example.src_sent))
                    print("Prolog Candidate")
                    print(hyp.to_prolog_template)
                    print("Prolog Reference")
                    print(example.to_prolog_template)
                    print("=======================================================")
                    """
                    return False
            elif self.args.lang == "geo_lambda" or self.args.lang == "atis_lambda" or self.args.lang == "overnight_lambda":
                if hyp.to_lambda_template == example.to_lambda_template:
                    return True
                else:
                    """
                    print("=======================================================")
                    print("Source Sentence")
                    print(" ".join(example.src_sent))
                    print("Lambda Candidate")
                    print(hyp.to_lambda_template)
                    print("Lambda Reference")
                    print(example.to_lambda_template)
                    print("=======================================================")
                    """
                    return False

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        correct_array = []
        oracle_array = []
        count = 0

        candidate_list = []
        reference_list = []

        reference_template_list = []

        src_list = []

        action_list = []

        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    """
                    if self.args.validate_grammar:
                        if self.args.lang.endswith('lambda'):
                            tgt_code = hyp_list[0].to_lambda_template
                        elif self.args.lang.endswith('prolog'):
                            tgt_code = hyp_list[0].to_prolog_template
                        if (not (validate_logic_form(tgt_code, self.grammar, self.transition_system))) and hyp_id == idx:
                            idx += 1
                    """

                    is_correct = self.is_hyp_correct(example, hyp)

                    hyp.is_correct = is_correct

                correct_array.append(hyp_list[0].is_correct)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
                if self.args.lang.endswith('lambda'):
                    candidate_list.append(hyp_list[0].to_lambda_template)
                elif self.args.lang.endswith('prolog'):
                    candidate_list.append(hyp_list[0].to_prolog_template)
                append_list(example, hyp, reference_list, reference_template_list, src_list, action_list,
                            data_type=self.args.lang)
            else:
                candidate_list.append("")
                append_list(example, hyp, reference_list, reference_template_list, src_list, action_list,
                            data_type=self.args.lang)

                correct_array.append(False)
                oracle_array.append(False)
        print("writing to file .................")
        file = None
        if self.args.dev_file is not None:
            file = self.args.dev_file
        elif self.args.test_file is not None:
            file = self.args.test_file
        write_to_file(file, candidate_list, reference_list, reference_template_list, src_list, action_list,
                      self.args.lang)

        acc = np.average(correct_array)

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc, correct_num = np.sum(correct_array))
        # print("Count : ", count)
        return eval_results


def append_list(example, hyp, reference_list, reference_template_list, src_list, action_list, data_type='lambda'):
    reference_list.append(example.to_logic_form)
    if data_type.endswith('lambda'):
        # print (reference_template_list)
        reference_template_list.append(example.to_lambda_template)
    elif data_type.endswith('prolog'):
        reference_template_list.append(example.to_prolog_template)
    src_list.append(" ".join(example.src_sent))
    action_list.append(" ".join([str(action) for action in hyp.actions]))


@Registrable.register('action_evaluator')
class ActionEvaluator(object):
    def __init__(self, args=None, vocab=None):
        self.precision = 'precision'
        self.recall = 'recall'
        self.f_measure = 'f_measure'
        self.support = 'support'
        self.action_vocab = vocab.action
        self.args = args

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        eval_results = {}

        for example, hyp_list in zip(examples, decode_results):
            ground_truth = set(example.tgt_actions)
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    pred = set(hyp.actions)
                    for action in pred:
                        if action not in eval_results:
                            eval_results[action] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}

                        if action in ground_truth:
                            eval_results[action]["tp"] = eval_results[action]["tp"] + 1
                        else:

                            eval_results[action]["fp"] = eval_results[action]["fp"] + 1

                    for action in ground_truth:

                        if action not in eval_results:
                            eval_results[action] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
                        eval_results[action]["support"] = eval_results[action]["support"] + 1
                        if action not in pred:
                            eval_results[action]["fn"] = eval_results[action]["fn"] + 1

        for action in eval_results:
            eval_results[action]["precision"] = eval_results[action]["tp"] / (
                    eval_results[action]["tp"] + eval_results[action]["fp"]) if (eval_results[action]["tp"] +
                                                                                 eval_results[action][
                                                                                     "fp"]) > 0 else 0
            eval_results[action]["recall"] = eval_results[action]["tp"] / (
                    eval_results[action]["tp"] + eval_results[action]["fn"]) if (
                                                                                        eval_results[action]["tp"] +
                                                                                        eval_results[action][
                                                                                            "fn"]) > 0 else 0
            eval_results[action]["f_measure"] = 2 * eval_results[action]["precision"] * eval_results[action][
                "recall"] / (
                                                        eval_results[action]["precision"] + eval_results[action][
                                                    "recall"]) if (
                                                                          eval_results[action]["precision"] +
                                                                          eval_results[action]["recall"]) > 0 else 0

        return eval_results


@Registrable.register('predicate_evaluator')
class ActionEvaluator(object):
    def __init__(self, args=None, vocab=None):
        self.precision = 'precision'
        self.recall = 'recall'
        self.f_measure = 'f_measure'
        self.support = 'support'
        self.action_vocab = vocab.action
        self.args = args

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):

        eval_results = {}

        for example, hyp_list in zip(examples, decode_results):
            ground_truth = set()
            # get gold logic form tokens
            gold_lf_list = example.to_logic_form.split(' ')
            for lf_token in gold_lf_list:
                if is_predicate(lf_token, dataset=self.args.lang):
                    ground_truth.add(lf_token)

            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):

                    if self.args.lang.endswith('lambda'):
                        candidate_lf_list = hyp.to_lambda_template.split(' ')
                    elif self.args.lang.endswith('prolog'):
                        candidate_lf_list = hyp.to_prolog_template.split(' ')
                    pred = set()
                    for can_token in candidate_lf_list:
                        if is_predicate(can_token, dataset=self.args.lang):
                            pred.add(can_token)

                    for action in pred:
                        if action not in eval_results:
                            eval_results[action] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}

                        if action in ground_truth:
                            eval_results[action]["tp"] = eval_results[action]["tp"] + 1
                        else:
                            eval_results[action]["fp"] = eval_results[action]["fp"] + 1

                    for action in ground_truth:

                        if action not in eval_results:
                            eval_results[action] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}

                        eval_results[action]["support"] = eval_results[action]["support"] + 1
                        if action not in pred:
                            eval_results[action]["fn"] = eval_results[action]["fn"] + 1
        # complete statistic
        eval_results['whole'] = {"precision": 0, "recall": 0, "f_measure": 0}
        for action in eval_results:
            if action == 'whole':
                continue
            eval_results[action]["precision"] = eval_results[action]["tp"] / (
                    eval_results[action]["tp"] + eval_results[action]["fp"]) if (eval_results[action]["tp"] +
                                                                                 eval_results[action][
                                                                                     "fp"]) > 0 else 0

            eval_results['whole']["precision"] = eval_results['whole']["precision"] + eval_results[action]["precision"]

            eval_results[action]["recall"] = eval_results[action]["tp"] / (
                    eval_results[action]["tp"] + eval_results[action]["fn"]) if (
                                                                                        eval_results[action]["tp"] +
                                                                                        eval_results[action][
                                                                                            "fn"]) > 0 else 0

            eval_results['whole']["recall"] = eval_results['whole']["recall"] + eval_results[action]["recall"]

            eval_results[action]["f_measure"] = 2 * eval_results[action]["precision"] * eval_results[action][
                "recall"] / (
                                                        eval_results[action]["precision"] + eval_results[action][
                                                    "recall"]) if (
                                                                          eval_results[action]["precision"] +
                                                                          eval_results[action]["recall"]) > 0 else 0

            eval_results['whole']["f_measure"] = eval_results['whole']["f_measure"] + eval_results[action]["f_measure"]

        eval_results['whole']["precision"] = eval_results['whole']["precision"] / (len(eval_results) - 1)
        eval_results['whole']["recall"] = eval_results['whole']["recall"] / (len(eval_results) - 1)
        eval_results['whole']["f_measure"] = eval_results['whole']["f_measure"] / (len(eval_results) - 1)

        return eval_results

@Registrable.register('smatch_evaluator')
class SmatchEvaluator(object):
    def __init__(self, args=None):
        self.args = args
        self.default_metric = 'accuracy'
        self.correct_num = 'correct_num'

    def get_triple_set(self, root):

        triple_set = []

        visited, queue = set(), [root]

        while queue:
            vertex = queue.pop(0)
            v_id = id(vertex)
            visited.add(v_id)
            current_triple = vertex.copy_no_link()
            for child in vertex.children:
                current_triple.add(child.copy_no_link())
                if id(child) not in visited:
                    queue.append(child)
            if len(vertex.children) > 0:
                triple_set.append(current_triple.to_lambda_expr)
            #print (current_triple)

        return triple_set

    def smatch_score(self, hyp_triple_set, exp_triple_set):
        correct_num = 0
        p = 0.
        r = 0.
        f = 0.
        #print (hyp_triple_set)
        #print (exp_triple_set)
        for triple in hyp_triple_set:
            if triple in exp_triple_set:
                correct_num += 1
        if len(exp_triple_set) == 0:
            p = 0
        else:
            p = correct_num / len(exp_triple_set)
        if len(hyp_triple_set) == 0:
            r = 0
        else:
            r = correct_num/len(hyp_triple_set)
        if p:
            f = 2*p*r/(p+r)

        return p, r, f

    def parse_prolog_query(self, elem_list):
        # ( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.filter ( var s ) ( string num_blocks ) ( string ! = ) ( number 3 block ) ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )	overnight-basketball
        # elem_list = ['job', '(', 'ANS', ')', ',', 'language', '(', 'ANS', ',', 'languageid0', ')', ',', '\\+', 'rex_exp', '(', 'area', '(', 'ANS', ',', 'areaid0', ')', ',', 'req_exp', '(', 'ANS', ')', ')', ';', '(', 'area', '(', 'ANS', ',', 'areaid1', ')', ',', 'req_exp', '(', 'ANS', ')', ')']
        separator_set = set([',', '(', ')', ';', '\+'])
        root = RuleVertex(ROOT)
        root.is_auto_nt = True
        root.position = 0
        depth = 0
        i = 0
        current = root
        node_pos = 1

        for elem in elem_list:
            # print("{} : {} ".format(elem, current.head))
            if elem == '(':
                depth += 1
                if i > 0:
                    last_elem = elem_list[i - 1]
                    # if last_elem == '\+':
                    # plus_flag = False

                    if last_elem in separator_set:
                        if last_elem == '\+':
                            depth += 1
                            current = current.children[-1]
                        child = RuleVertex(IMPLICIT_HEAD)
                        child.parent = current
                        current.add(child)
                        child.is_auto_nt = True

                        current = child
                        child.position = node_pos
                        node_pos += 1

                    else:
                        current = current.children[-1]

            elif elem == ')':
                if current is None:
                    break
                current = current.parent
                depth -= 1
                if current and current.head == '\+':
                    current = current.parent
                    depth -= 1
                # plus_flag = True
            elif not elem == ',':
                if i > 0:
                    last_elem = elem_list[i - 1]
                    if last_elem == '\+':
                        depth += 1
                        current = current.children[-1]

                norm_elem = elem
                child = RuleVertex(norm_elem)
                child.parent = current
                current.add(child)
                child.position = node_pos
                node_pos += 1
            i += 1

        return root

    def parse_lambda_query(self, elem_list):
        root = RuleVertex(ROOT)
        root.is_auto_nt = True
        root.position = 0
        i = 0
        current = root
        node_pos = 1
        var_id = 0
        for elem in elem_list:
            if current is None:
                break
            if elem == ')':
                current = current.parent
            elif not elem in [',', '(', ';']:
                last_elem = elem_list[i - 1]
                if last_elem == '(':
                    child = RuleVertex(elem)
                    child.parent = current
                    current.add(child)
                    current = child
                    child.position = node_pos
                    node_pos += 1
                else:
                    norm_elem = elem
                    child = RuleVertex(norm_elem)

                    child.parent = current
                    current.add(child)
                    child.position = node_pos
                    node_pos += 1
            i += 1

        return root

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):

        correct_array = []

        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    if hyp.to_lambda_template or hyp.to_logic_form:
                        logical_form = hyp.to_logic_form
                        if self.args.lang.endswith('prolog'):
                            logical_form = '( ' + hyp.to_logic_form + ' )'
                        if self.args.parser == 'seq2seq' or self.args.parser == 'seq2seq_dong':
                            print (logical_form)
                            if self.args.lang.endswith('lambda'):
                                if self.args.lang == 'geo_lambda' or self.args.lang == 'atis_lambda':
                                    hyp_node = self.parse_lambda_query(logical_form.split(' '))
                                    #print (hyp_node)
                                    example_node = self.parse_lambda_query(example.to_logic_form.split(' '))
                                    #print (example_node)
                                elif self.args.lang == 'overnight_lambda':
                                    hyp_node = parse_overnight_query_helper(logical_form.split(' '))
                                    #print (hyp_node)
                                    example_node = parse_overnight_query_helper(example.to_logic_form.split(' '))
                            elif self.args.lang.endswith('prolog'):
                                hyp_node = self.parse_prolog_query(logical_form.split(' '))
                                example_node = self.parse_prolog_query(example.to_logic_form.split(' '))
                        else:
                            #print ("==========")
                            #print (hyp.to_lambda_template)
                            if self.args.lang.endswith('lambda'):
                                if self.args.lang == 'geo_lambda' or self.args.lang == 'atis_lambda':
                                    hyp_node = self.parse_lambda_query(hyp.to_lambda_template.split(' '))
                                    #print (hyp_node)
                                    example_node = self.parse_lambda_query(example.to_logic_form.split(' '))
                                    #print (example_node)
                                elif self.args.lang == 'overnight_lambda':
                                    hyp_node = parse_overnight_query_helper(hyp.to_lambda_template.split(' '))
                                    #print (hyp_node)
                                    example_node = parse_overnight_query_helper(example.to_logic_form.split(' '))
                            elif self.args.lang.endswith('prolog'):
                                hyp_node = self.parse_prolog_query(hyp.to_prolog_template.split(' '))
                                example_node = self.parse_prolog_query(example.to_logic_form.split(' '))

                        hyp_triple_set = self.get_triple_set(hyp_node)
                        #print (hyp_triple_set)
                        exp_triple_set = self.get_triple_set(example_node)
                        p, r, f = self.smatch_score(hyp_triple_set, exp_triple_set)
                        hyp.score = f
                    else:
                        hyp.score = 0.
                correct_array.append(hyp_list[0].score)
            else:
                correct_array.append(0.)
        #print (correct_array)
        acc = np.average(correct_array)

        eval_results = dict(accuracy=acc, correct_num=np.sum(correct_array))
        # print("Count : ", count)
        return eval_results


@Registrable.register('denotation_evaluator')
class DenotationEvaluator(object):
    def __init__(self, args=None):
        self.args = args
        self.default_metric = 'accuracy'
        self.correct_num = 'correct_num'

    def evaluate_dataset(self, examples, decode_results, fast_mode=False, test_data = None):

        correct_array = [None]*len(examples)

        if test_data.known_domains:
            current_domain = test_data.known_domains[0]
        else:
            current_domain = test_data.domain

        odomain = OvernightDomain(current_domain)
        idx = 0
        length_indx = 0
        current_domain_thres = test_data.known_test_data_length[0]
        # print (current_domain)
        # print (current_domain_thres)
        hyp_lf_list = {}
        exp_lf_list = {}
        for example, hyp_list in zip(examples, decode_results):
            if idx >= current_domain_thres:
                hyp_lf_values = hyp_lf_list.values()
                exp_lf_values = exp_lf_list.values()

                hyp_lf_normalized = odomain.normalize(hyp_lf_values)

                example_lf_normalized = odomain.normalize(exp_lf_values)

                hyp_denotatations = odomain.obtain_denotations(hyp_lf_normalized)

                example_denotatations = odomain.obtain_denotations(example_lf_normalized)

                sorted_hyp_lf_list = sorted(hyp_lf_list.items())
                in_idx = 0
                for hyp_idx, hyp_lf in sorted_hyp_lf_list:
                    if hyp_denotatations[in_idx] == example_denotatations[in_idx]:
                        correct_array[hyp_idx] = True
                    else:
                        correct_array[hyp_idx] = False

                    in_idx += 1

                length_indx += 1
                current_domain_thres = test_data.known_test_data_length[length_indx]
                current_domain = test_data.known_domains[length_indx]
                odomain = OvernightDomain(current_domain)
                hyp_lf_list = {}
                exp_lf_list = {}

            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                hyp = hyp_list[0]
                if hyp.to_lambda_template or hyp.to_logic_form:
                    hyp_logical_form = hyp.to_lambda_template
                    hyp_lf_list[idx] = hyp_logical_form
                    example_logical_form = example.to_logic_form
                    exp_lf_list[idx] = example_logical_form

                    #hyp_lf_normalized = odomain.normalize([hyp_logical_form])

                    #example_lf_normalized = odomain.normalize([example_logical_form])

                    #hyp_denotatations = odomain.obtain_denotations(hyp_lf_normalized)

                    #example_denotatations = odomain.obtain_denotations(example_lf_normalized)
                    # print (example_denotatations)
                else:
                    correct_array[idx] = False

            else:
                correct_array[idx] = False

            idx += 1

        hyp_lf_values = hyp_lf_list.values()
        exp_lf_values = exp_lf_list.values()

        hyp_lf_normalized = odomain.normalize(hyp_lf_values)

        example_lf_normalized = odomain.normalize(exp_lf_values)

        hyp_denotatations = odomain.obtain_denotations(hyp_lf_normalized)

        example_denotatations = odomain.obtain_denotations(example_lf_normalized)

        sorted_hyp_lf_list = sorted(hyp_lf_list.items())
        in_idx = 0
        for hyp_idx, hyp_lf in sorted_hyp_lf_list:
            # print(hyp_denotatations[in_idx])
            if hyp_denotatations[in_idx] == example_denotatations[in_idx]:
                correct_array[hyp_idx] = True
            else:
                correct_array[hyp_idx] = False

            in_idx += 1
        #print (correct_array)
        acc = np.average(correct_array)

        eval_results = dict(accuracy=acc, correct_num = np.sum(correct_array))

        return eval_results