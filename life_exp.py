# coding=utf-8
from __future__ import print_function

import argparse
import uuid
import datetime
from itertools import chain

import six.moves.cPickle as pickle
from six.moves import xrange as range
from six.moves import input
import traceback

import numpy as np
import time
import os
import sys
import collections
import torch
from torch.autograd import Variable

from continual_model.utils import confusion_matrix
from model.utils import GloveHelper, merge_vocab_entry
from common.registerable import Registrable
from components.dataset import Dataset, Example, ContinuumDataset
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils
import evaluation
from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.evaluator import DefaultEvaluator, ActionEvaluator, SmatchEvaluator
from continual_model.gem import Net
from continual_model.icarl import Net
from continual_model.emr import Net
from continual_model.ea_emr import Net
from continual_model.origin import Net
from continual_model.independent import Net
from continual_model.gem_emr import Net
from continual_model.loss_emr import Net
from continual_model.loss_gem import Net
from continual_model.a_gem import Net
from continual_model.gss_emr import Net
from continual_model.emr_wo_task_boundary import Net
from continual_model.adap_emr import Net
# from model.seq2seq_align import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    return args

def init_parameter_seed():
    # seed the RNG
    torch.manual_seed(args.p_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.p_seed)
    np.random.seed(int(args.p_seed * 13 / 7))


def read_domain_data(domains, file_prefix, suffix):
    path_list = []
    for domain in domains:
        path = os.path.join(file_prefix, domain + suffix)
        path_list.append(path)
    return path_list


def read_domain_vocab(domains, file_prefix, suffix):
    assert len(domains) > 0
    vocab = pickle.load(open(os.path.join(file_prefix, domains[0] + suffix), 'rb'))
    for domain in domains[1:]:
        vocab_after = pickle.load(open(os.path.join(file_prefix, domain + suffix), 'rb'))
        vocab = merge_vocab_entry(vocab_after, vocab)

    return vocab


def eval_tasks(model, tasks, args):
    was_training = model.training
    model.eval()
    result = []
    correct_num = 0
    total_num = 0
    for i, task in enumerate(tasks):
        decode_results = model.decode(task.test.examples, i)

        if args.evaluator == "denotation_evaluator":
            eval_results = model.evaluator.evaluate_dataset(task.test.examples, decode_results,
                                                           fast_mode=args.eval_top_pred_only, test_data = task.test)
        else:
            eval_results = model.evaluator.evaluate_dataset(task.test.examples, decode_results,
                                                           fast_mode=args.eval_top_pred_only)
        print(eval_results, file=sys.stderr)
        test_score = eval_results[model.evaluator.default_metric]

        result.append(test_score)
        total_num += len(task.test.examples)
        correct_num += eval_results[model.evaluator.correct_num]
    final_acc = correct_num/total_num
    result.append(final_acc)
    if was_training: model.train()
    return result

def life_experience(net, continuum_dataset, args):
    result_a = []

    time_start = time.time()

    for task_indx, task_data in enumerate(continuum_dataset):
        result_a.append(eval_tasks(net, continuum_dataset, args))
        net.train()
        net.observe(task_data, task_indx)

    result_a.append(eval_tasks(net, continuum_dataset, args))


    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_a), time_spent

if __name__ == '__main__':

    arg_parser = init_arg_parser()
    args = init_config()

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    print(args, file=sys.stderr)
    domains = args.domains
    print("training started ...")

    train_path_list = read_domain_data(domains, args.train_file, "_train.bin")

    test_path_list = read_domain_data(domains, args.train_file, "_test.bin")

    if args.dev_file:
        dev_path_list = read_domain_data(domains, args.dev_file, "_dev.bin")
    else:
        dev_path_list = None

    vocab_path_list = read_domain_data(domains, args.vocab, ".vocab.freq.bin")

    continuum_dataset = ContinuumDataset.read_continuum_data(args, train_path_list, test_path_list, dev_path_list, vocab_path_list, domains)
    init_parameter_seed()
    # unique identifier
    uid = uuid.uuid4().hex

    print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    net = parser_cls(args, continuum_dataset)
    result_a, spent_time = life_experience(net, continuum_dataset, args)
    save_dir = os.path.join(args.save_decode_to, args.parser)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fname = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_parser.' + str(args.parser) + \
             '_max_ep.' + str(args.max_epoch) + \
             '_ratio.' + str(args.num_exemplars_ratio) + \
             '_samp.' + str(args.sample_method) + \
             '_lr.' + str(args.lr) + \
             '_perm_sed.' + str(args.seed) + \
             '_para_sed.' + str(args.p_seed) + \
             '_subs.' + str(args.subselect) + \
             '_reba.' + str(args.rebalance) + \
             '_bat.' + str(args.batch_size) + \
             '_nd.' + str(args.num_known_domains) + \
             '_e_num.' + str(args.num_exemplars_per_task) + \
             '_sm.' + str(args.reg) + \
             '_ewc.' + str(args.ewc)

    fname = os.path.join(save_dir, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))