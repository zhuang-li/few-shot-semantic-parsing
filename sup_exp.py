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

import torch
from torch.autograd import Variable
from model.utils import GloveHelper
from common.registerable import Registrable
from components.dataset import Dataset, Example
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils
import evaluation
from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.evaluator import DefaultEvaluator, ActionEvaluator
from model.seq2seq_cross_templates import Seq2SeqModel
from model.seq2seq_dong import Seq2SeqModel
from model.seq2seq_batch_action import Seq2SeqModel
from model.seq2seq_c_t_stack import Seq2SeqModel
from model.seq2seq_action import Seq2SeqModel
from model.seq2seq_bi_action import Seq2SeqModel
from model.seq2seq_template import Seq2SeqModel
from model.seq2seq_topdown import Seq2SeqModel
from model.ratsql import Seq2SeqModel
from model.irnet import Seq2SeqModel
#from model.seq2seq_align import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def train(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    print ("training started ...")
    train_set = Dataset.from_bin_file(args.train_file)
    train_set = Dataset(train_set.examples)
    for e in train_set.examples:
        e.meta = 0
    #+ train_set.examples[-6:] * 4
    if args.support_file:
        support_set = Dataset.from_bin_file(args.support_file)
        for e in support_set.examples:
            e.meta = 1
        if args.augment == 0:
            train_set = Dataset(train_set.examples + support_set.examples)
        elif args.augment == 1:
            train_set = Dataset(train_set.examples + support_set.examples*10)
        elif args.augment == 2:
            train_set = Dataset(train_set.examples + support_set.examples*5)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else:
        dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))

    print ("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    model = parser_cls(vocab, args)
    print ("setting model to training mode")
    model.train()

    evaluator = Registrable.by_name(args.evaluator)(args=args)
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
    report_loss = report_examples = report_sup_att_loss = 0.
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
            if loss.numel() > 1:
                #print ("loss size ", loss.size())
                loss_val = torch.sum(loss).data.item()
                #print(loss.size())
                loss = torch.mean(loss)
                #print("===================")
                #print (loss)
            elif loss.numel() == 1:
                #print ("1 loss size ", loss.size())
                loss_val = loss.item()
                loss = loss / len(batch_examples)
            else:
                raise ValueError
            report_loss += loss_val
            report_examples += len(batch_examples)

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs is not None:
                    #print (len(att_probs))
                    sup_att_loss = -torch.stack(att_probs).mean()
                    sup_att_loss_val = sup_att_loss.data.item()
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

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
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                """
                model_file = args.save_to + '.bin'
                print('save the current model ..', file=sys.stderr)
                print('save model to [%s]' % model_file, file=sys.stderr)
                model.save(model_file)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
                """
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
            model_file = args.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')

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
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
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
                #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    saved_args.use_cuda = args.use_cuda

    parser_cls = Registrable.by_name(args.parser)
    parser, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda,args=args)
    parser.eval()
    vocab = parser.vocab
    evaluator = Registrable.by_name(args.evaluator)(args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    if args.evaluator == 'action_evaluator':
        for action in eval_results:
            print ("{0} precision : {1:.4f}".format(action,eval_results[action][evaluator.precision]))
            print ("{0} recall : {1:.4f}".format(action,eval_results[action][evaluator.recall]))
            print ("{0} f measure : {1:.4f}".format(action,eval_results[action][evaluator.f_measure]))
            print ("{0} support : {1:d}".format(action,eval_results[action][evaluator.support]))
    elif args.evaluator == 'predicate_evaluator':
        for action in eval_results:
            if action == 'whole':
                continue
            print("{0} precision : {1:.4f}".format(action, eval_results[action][evaluator.precision]))
            print("{0} recall : {1:.4f}".format(action, eval_results[action][evaluator.recall]))
            print("{0} f measure : {1:.4f}".format(action, eval_results[action][evaluator.f_measure]))
            print("{0} support : {1:d}".format(action, eval_results[action][evaluator.support]))
        print("{0} precision : {1:.4f}".format("whole", eval_results["whole"][evaluator.precision]))
        print("{0} recall : {1:.4f}".format("whole", eval_results["whole"][evaluator.recall]))
        print("{0} f measure : {1:.4f}".format("whole", eval_results["whole"][evaluator.f_measure]))
    else:
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
    else:
        raise RuntimeError('unknown mode')