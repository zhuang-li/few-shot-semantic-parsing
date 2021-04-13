# coding=utf-8
from __future__ import print_function

import argparse
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
from model.utils import GloveHelper, merge_vocab_entry
from common.registerable import Registrable
from components.dataset import Dataset, Example, ContinualDataset
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils
import evaluation
from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.evaluator import DefaultEvaluator, ActionEvaluator
from continual_model.seq2seq_topdown import Seq2SeqModel


# from model.seq2seq_align import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def continual_train(args):
    known_domains = args.known_domains
    new_domains = args.new_domains
    model_path = args.load_model
    last_domain = 'sample'
    for domain in new_domains:
        train_a_task(args, known_domains, [domain], model_path)
        known_domains.append(domain)
        model_path = str(args.load_model).replace(last_domain, domain)
        last_domain = domain


def train_iter_process(optimizer, batch_examples, report_loss, report_examples, model):
    optimizer.zero_grad()

    ret_val = model(batch_examples)

    loss = ret_val[0]

    loss_val = torch.sum(loss).data.item()

    loss = torch.mean(loss)

    report_loss += loss_val
    report_examples += len(batch_examples)

    loss.backward()

    # clip gradient
    if args.clip_grad > 0.:
        if args.clip_grad_mode == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        elif args.clip_grad_mode == 'value':
            grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

    optimizer.step()

    return report_loss, report_examples


def train_a_task(args, known_domains, new_domain, model_path):
    """Maximum Likelihood Estimation"""

    previous_train_path_list = read_domain_data(known_domains, args.train_file, "_train.bin")

    known_train_set = dict()
    if args.sample == 'task_level':
        idx = 0
        for path in previous_train_path_list:
            domain = known_domains[idx]
            task_set = ContinualDataset.from_bin_file(path)
            known_train_set[domain] = task_set
            idx += 1
    elif args.sample == 'sample_level':
        known_train_set['sample'] = ContinualDataset.from_bin_file_list(previous_train_path_list)

    print("training started ...")

    train_path_list = read_domain_data(new_domain, args.train_file, "_train.bin")
    train_set = ContinualDataset.from_bin_file_list(train_path_list)

    if args.dev_file:

        dev_path_list = read_domain_data(new_domain, args.dev_file, "_test.bin")
        dev_set = ContinualDataset.from_bin_file_list(dev_path_list)

    else:
        dev_set = ContinualDataset(examples=[])

    vocab = read_domain_vocab(known_domains + new_domain, args.vocab, ".vocab.freq.bin")

    print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    model, src_vocab = parser_cls.load(model_path=model_path, use_cuda=args.use_cuda, loaded_vocab=vocab, args=args)
    print("load pre-trained word embedding (optional)")
    if args.glove_embed_path and (src_vocab is not None):
        # print (args.embed_size)
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
        glove_embedding.load_pre_train_to(model.src_embed, src_vocab)

    print("setting model to training mode")
    model.train()

    evaluator = Registrable.by_name(args.evaluator)(args=args, vocab=vocab)
    if args.use_cuda and torch.cuda.is_available():
        model.cuda()
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    if args.optimizer == 'RMSprop':
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    elif args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    print("load pre-trained word embedding (optional)")
    if args.glove_embed_path:
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
        glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = replay_report_loss = replay_report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            train_iter += 1
            # normal train
            report_loss, report_examples = train_iter_process(optimizer, batch_examples, report_loss, report_examples,
                                                              model)
            if args.sample == 'sample_level':
                if args.sample_mode == 'random':
                    replay_batch_examples = known_train_set['sample'].random_sample_batch_iter(args.batch_size)
                elif args.sample_mode == 'template':
                    replay_batch_examples = known_train_set['sample'].template_sample_batch_iter(args.batch_size, batch_examples)
                else:
                    raise ValueError
                replay_report_loss, replay_report_examples = train_iter_process(optimizer, replay_batch_examples,
                                                                                replay_report_loss,
                                                                                replay_report_examples, model)
            elif args.sample == 'task_level':
                for domain, domain_set in known_train_set.items():
                    sample_size = int(args.batch_size / len(known_train_set))
                    if args.sample_mode == 'random':
                        replay_task_examples = domain_set.random_sample_batch_iter(sample_size)
                    elif args.sample_mode == 'template':
                        replay_task_examples = domain_set.template_sample_batch_iter(sample_size, batch_examples)
                    else:
                        raise ValueError
                    replay_report_loss, replay_report_examples = train_iter_process(optimizer, replay_task_examples,
                                                                                    replay_report_loss,
                                                                                    replay_report_examples, model)
            # replay train
            # replay_report_loss, replay_report_examples = train_iter_process(optimizer, replay_batch_examples, replay_report_loss, replay_report_examples, model)

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if not args.sample == 'normal_level':
                    log_str += '[Iter %d] replay encoder loss=%.5f' % (
                    train_iter, replay_report_loss / replay_report_examples)

                print(log_str, file=sys.stderr)
                report_loss = report_examples = replay_report_loss = replay_report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.%s.iter%d.bin' % (new_domain[0], train_iter)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                    epoch, eval_results,
                    evaluator.default_metric,
                    dev_score,
                    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:

            patience = 0
            model_file = args.save_to + '.%s.bin' % new_domain[0]
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.%s.optim.bin' % new_domain[0])

        elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            break

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                break

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.%s.bin' % new_domain[0], map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.use_cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                reset_optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
                if args.optimizer == 'RMSprop':
                    optimizer = reset_optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
                else:
                    optimizer = reset_optimizer_cls(model.parameters(), lr=args.lr)
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.%s.optim.bin' % new_domain[0]))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


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


def train(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set

    new_domains = args.new_domains
    print("training started ...")

    train_path_list = read_domain_data(new_domains, args.train_file, "_train.bin")

    train_set = ContinualDataset.from_bin_file_list(train_path_list)

    if args.dev_file:
        dev_path_list = read_domain_data(new_domains, args.dev_file, "_test.bin")
        dev_set = ContinualDataset.from_bin_file_list(dev_path_list)
    else:
        dev_set = ContinualDataset(examples=[])

    vocab = read_domain_vocab(new_domains, args.vocab, ".vocab.freq.bin")

    print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    model = parser_cls(vocab, args)
    print("setting model to training mode")
    model.train()

    evaluator = Registrable.by_name(args.evaluator)(args=args, vocab=vocab)
    if args.use_cuda and torch.cuda.is_available():
        model.cuda()
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    if args.optimizer == 'RMSprop':
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    elif args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    print("load pre-trained word embedding (optional)")
    if args.glove_embed_path:
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
        glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model(batch_examples)

            loss = ret_val[0]

            loss_val = torch.sum(loss).data.item()
            # print(loss.size())
            loss = torch.mean(loss)

            report_loss += loss_val
            report_examples += len(batch_examples)

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                if args.clip_grad_mode == 'norm':
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                elif args.clip_grad_mode == 'value':
                    grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.sample.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                    epoch, eval_results,
                    evaluator.default_metric,
                    dev_score,
                    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:

            patience = 0
            model_file = args.save_to + '.sample.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.sample.optim.bin')

        elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.sample.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.use_cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                reset_optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
                if args.optimizer == 'RMSprop':
                    optimizer = reset_optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
                else:
                    optimizer = reset_optimizer_cls(model.parameters(), lr=args.lr)
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.sample.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args):
    domains = args.new_domains
    test_path_list = read_domain_data(domains, args.test_file, "_test.bin")
    # print (test_path_list)
    test_set = ContinualDataset.from_bin_file_list(test_path_list)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    saved_args.use_cuda = args.use_cuda

    parser_cls = Registrable.by_name(args.parser)
    parser, unit_src_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda)
    parser.eval()
    vocab = parser.vocab
    evaluator = Registrable.by_name(args.evaluator)(args=args, vocab=vocab)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)

    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'continual_train':
        continual_train(args)
    else:
        raise RuntimeError('unknown mode')
