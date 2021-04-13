# coding=utf-8
from __future__ import print_function

import argparse
from collections import OrderedDict
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
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from torchviz import make_dot

from model.nn_utils import direct_clip_grad_norm_
from model.utils import GloveHelper
from common.registerable import Registrable
from components.dataset import Dataset, Example, EposideDataset
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils
import evaluation
from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.evaluator import DefaultEvaluator, ActionEvaluator
from model.seq2seq_cross_templates import Seq2SeqModel
from model.seq2seq_dong import Seq2SeqModel
from model.seq2seq_c_t_stack import Seq2SeqModel
from model.seq2seq_action import Seq2SeqModel
from model.seq2seq_bi_action import Seq2SeqModel
from model.seq2seq_template import Seq2SeqModel
from model.seq2seq_few_shot import Seq2SeqModel


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def copy_model_and_optimizer_param(model, new_model):
    new_model.load_state_dict(model.state_dict())  # copy state
    # new_optimizer.load_state_dict(optimizer.state_dict())  # copy state

    return new_model


def forward_pass(model, examples, args, weight=None):
    if weight is None:
        sup_att_loss_val = 0
        ret_val = model(examples)

        loss = ret_val[0]
        if loss.numel() > 1:
            # print ("loss size ", loss.size())
            loss_val = torch.sum(loss).data.item()
            loss = torch.mean(loss)
        elif loss.numel() == 1:
            # print ("1 loss size ", loss.size())
            loss_val = loss.item()
            loss = loss / len(examples)
        else:
            raise ValueError
        # report_loss += loss_val
        # report_examples += len(examples)

        if args.sup_attention:
            att_probs = ret_val[1]
            if att_probs:
                # print (len(att_probs))
                sup_att_loss = -torch.stack(att_probs).mean()
                sup_att_loss_val = sup_att_loss.data.item()
                # report_sup_att_loss += sup_att_loss_val

                loss += sup_att_loss
    else:
        sup_att_loss_val = 0
        ret_val = model(examples, weight)

        loss = ret_val[0]
        if loss.numel() > 1:
            # print ("loss size ", loss.size())
            loss_val = torch.sum(loss).data.item()
            loss = torch.mean(loss)
        elif loss.numel() == 1:
            # print ("1 loss size ", loss.size())
            loss_val = loss.item()
            loss = loss / len(examples)
        else:
            raise ValueError
        # report_loss += loss_val
        # report_examples += len(examples)

        if args.sup_attention:
            att_probs = ret_val[1]
            if att_probs:
                # print (len(att_probs))
                sup_att_loss = -torch.stack(att_probs).mean()
                sup_att_loss_val = sup_att_loss.data.item()
                # report_sup_att_loss += sup_att_loss_val

                loss += sup_att_loss
    """
    optimizer.zero_grad()
    if backward:
        loss.backward()

        # clip gradient
        if args.clip_grad > 0.:
            if args.clip_grad_mode == 'norm':
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            elif args.clip_grad_mode == 'value':
                grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
    """
    return loss_val, sup_att_loss_val, loss


def assign_model_parameters(model, temp_parameters):
    idx = 0
    assert len(temp_parameters) == len(list(model.parameters()))
    for param in model.parameters():
        param.data.copy_(temp_parameters[idx].data)
        idx += 1


def clone_model_parameters(model):
    return [w.clone() for w in list(model.parameters())]


def update_parameters(temp_parameters, loss, alpha, clip_grad=5.0):
    weights = temp_parameters
    grads = torch.autograd.grad(loss, weights, allow_unused=True)
    #print (grads)
    direct_clip_grad_norm_(grads, clip_grad)
    #print (grads)
    new_weights = []
    idx = 0
    for w, g in zip(weights, grads):
        if g is not None:
            #print (g)
            #print (idx)
            new_weights.append(w - alpha * g.detach())
        else:
            new_weights.append(w)
        idx += 1
    assert len(new_weights) == len(weights)
    return new_weights


def copy_gradient(model, inner_model):
    for paramName, paramValue, in inner_model.named_parameters():
        for netCopyName, netCopyValue, in model.named_parameters():
            if paramName == netCopyName:
                netCopyValue.grad = paramValue.grad.clone()


def fine_tune(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else:
        dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))

    print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg

    print("fine-tuning started ...")
    print("========================================================================================")
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    # pre_trained_parser = parser_cls.load(model_path=args.load_model, use_cuda=False, mode=args.few_shot_mode)
    model, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda,
                                                     loaded_vocab=vocab, args=args)

    print("setting model to training mode")

    model.train()

    evaluator = Registrable.by_name(args.evaluator)(args=args)
    if args.use_cuda and torch.cuda.is_available():
        model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    if args.optimizer == 'RMSprop':
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
        # if args.few_shot_mode == 'pre_train':
        # inner_optimizer = optimizer_cls(inner_model.parameters(), lr=args.lr, alpha=args.alpha)  # get new optimiser
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
        # if args.few_shot_mode == 'pre_train':
        # inner_optimizer = SGD(inner_model.parameters(), lr=0.05)  # get new optimiser

    print('begin fine-tuning, %d training examples' % (len(train_set)), file=sys.stderr)
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

            loss_val, sup_att_loss_val, loss = forward_pass(model, batch_examples, model.args)
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                if args.clip_grad_mode == 'norm':
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                elif args.clip_grad_mode == 'value':
                    grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
            optimizer.step()
            report_loss += loss_val
            report_examples += len(batch_examples)
            report_sup_att_loss += sup_att_loss_val
            # free the model and the optimizer

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

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch and epoch%args.valid_every_epoch == 0:
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
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def meta_update(model, opt, dummy_examples, ls):
    #print ('\n Meta update \n')
    # We use a dummy forward / backward pass to get the correct grads into self.net
    dummy_examples.sort(key=lambda e: -len(e.src_sent))
    dummy_loss_val, dummy_loss_val, loss = forward_pass(model, dummy_examples, model.args)
    # Unpack the list of grad dicts
    gradients = {k: sum(d[k] for d in ls if d[k] is not None) for k in ls[0].keys()}
    # Register a hook on each parameter in the net that replaces the current dummy grad
    # with our grads accumulated across the meta-batch
    hooks = []
    for (k,v) in model.named_parameters():
        def get_closure():
            key = k
            def replace_grad(grad):
                return gradients[key]
            return replace_grad
        hooks.append(v.register_hook(get_closure()))
    # Compute grads for current step, replace with summed gradients as defined by hook
    opt.zero_grad()
    loss.backward()

    if model.args.clip_grad > 0.:
        if model.args.clip_grad_mode == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.args.clip_grad)
        elif model.args.clip_grad_mode == 'value':
            grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), model.args.clip_grad)

    # Update the net parameters with the accumulated gradient according to optimizer
    opt.step()
    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

def pre_train(args):
    """Maximum Likelihood Estimation"""
    with torch.backends.cudnn.flags(enabled=False):
        # load in train/dev set
        train_set = EposideDataset.from_bin_file(args.train_file)
        if args.dev_file:
            dev_set = Dataset.from_bin_file(args.dev_file)
        else:
            dev_set = Dataset(examples=[])
        vocab = pickle.load(open(args.vocab, 'rb'))

        print("register parser ...")
        parser_cls = Registrable.by_name(args.parser)  # TODO: add arg

        print("pre-training started ...")
        model = parser_cls(vocab, args)

        print("setting model to training mode")

        model.train()

        evaluator = Registrable.by_name(args.evaluator)(args=args)
        if args.use_cuda and torch.cuda.is_available():
            model.cuda()

        optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
        if args.optimizer == 'RMSprop':
            optimizer = optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
            # if args.few_shot_mode == 'pre_train':
            # inner_optimizer = optimizer_cls(inner_model.parameters(), lr=args.lr, alpha=args.alpha)  # get new optimiser
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

        print('begin pre-training, %d training examples' % (len(train_set)), file=sys.stderr)
        print('vocab: %s' % repr(vocab), file=sys.stderr)

        epoch = train_iter = 0
        report_loss = report_examples = report_sup_att_loss = inner_report_loss = inner_report_examples = inner_report_sup_att_loss = 0.
        history_dev_scores = []
        num_trial = patience = 0

        while True:
            epoch += 1
            epoch_begin = time.time()
            #losses_q = 0
            model.zero_grad()
            # copy_model_and_optimizer_param(model, inner_model)
            # initial_parameters = clone_model_parameters(model)

            grad_list = []
            dummy_examples = []
            #print (initial_parameters[20])
            for support_examples, query_examples in train_set.task_batch_iter(n_way=args.n_way, k_shot=args.k_shot,
                                                                              query_num=args.query_num,
                                                                              task_num=args.task_num, shuffle=True):
                fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())
                dummy_examples.extend(support_examples)
                train_iter += 1
                # temp_parameters = list(model.parameters())
                #assign_model_parameters(model, temp_parameters)
                #print ("11111111111111111111111111")
                #print (list(model.parameters())[2])
                inner_loss_val, inner_sup_att_loss_val, inner_loss = forward_pass(model, support_examples, model.args)

                inner_grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, retain_graph=True, allow_unused=True)

                fast_weights = OrderedDict(
                    (name, param - args.step_size * grad) if grad is not None else (name, param) for ((name, param), grad) in zip(fast_weights.items(), inner_grads))

                # print (inner_loss)
                #print ("2222222222222222222222222")
                #print(temp_parameters[5])
                #print (temp_parameters[11])
                # temp_parameters = update_parameters(temp_parameters, inner_loss, args.step_size, clip_grad=args.clip_grad)

                #print (temp_parameters[11])

                # print (inner_optimizer.param_groups[0]['lr'])
                inner_report_loss += inner_loss_val
                inner_report_examples += len(support_examples)
                inner_report_sup_att_loss += inner_sup_att_loss_val
                # print("inner=====================================")
                #print ("33333333333333333333333333")
                #print(temp_parameters[5])
                # assign_model_parameters(model, temp_parameters)
                #print ("444444444444444444444444444444444")
                #print (list(model.parameters())[2])
                loss_val, sup_att_loss_val, loss = forward_pass(model, query_examples, model.args, fast_weights)
                #temp_parameters = list(model.parameters())
                #losses_q.append(loss)
                #losses_q += loss
                report_loss += loss_val
                report_examples += len(query_examples)
                report_sup_att_loss += sup_att_loss_val
                # free the model and the optimizer
                loss = loss / args.task_num
                grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
                #print ("=============")
                meta_grads = {name: g for ((name, _), g) in zip(model.named_parameters(), grads)}
                grad_list.append(meta_grads)
                #model.zero_grad()

                if train_iter % args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                    inner_log_str = '[Iter %d] inner_encoder loss=%.5f' % (
                    train_iter, inner_report_loss / inner_report_examples)
                    if args.sup_attention:
                        log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                        inner_log_str += ' supervised attention loss=%.5f' % (
                                    inner_report_sup_att_loss / inner_report_examples)
                        inner_report_sup_att_loss = 0.
                        report_sup_att_loss = 0.

                    print(log_str, file=sys.stderr)
                    print(inner_log_str, file=sys.stderr)
                    inner_report_loss = inner_report_examples = 0.
                    report_loss = report_examples = 0.

            meta_update(model, optimizer, dummy_examples, grad_list)
            #print (initial_parameters[20])
            # assign_model_parameters(model, initial_parameters)
            """
            model_parameters = list(model.parameters())
            for loss in losses_q:
                #print (make_dot(losses_q[i]))
                grads = torch.autograd.grad(loss, model_parameters, create_graph=True, retain_graph=False, allow_unused=True)
                #print (grads[0])
                for w, g in zip(model_parameters, grads):
                    if w.grad is not None:
                        if g is None:
                            #print ("1")
                            w.grad = w.grad
                        else:
                            #print ("2")
                            #print (w.grad)
                            w.grad = w.grad + g.detach()
                    else:
                        if g is None:
                            #print ("3")
                            w.grad = w.grad
                        else:
                            #print ("4")
                            w.grad = g.detach()
                """
                #loss.backward()

            #print (model_parameters[10])
            #print (model_parameters[10].grad)
            #print ()
                #optimizer.step()
            #print (model_parameters[10])
            #print(model_parameters[10].grad)

            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

            if args.save_all_models:
                model_file = args.save_to + '.iter%d.bin' % train_iter
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

            if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch and epoch%args.valid_every_epoch == 0:
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
                    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    parser, src_vocab,vertex = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda, args=args)
    parser.eval()
    vocab = parser.vocab
    evaluator = Registrable.by_name(args.evaluator)(args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    if args.evaluator == 'action_evaluator':
        for action in eval_results:
            print("{0} precision : {1:.4f}".format(action, eval_results[action][evaluator.precision]))
            print("{0} recall : {1:.4f}".format(action, eval_results[action][evaluator.recall]))
            print("{0} f measure : {1:.4f}".format(action, eval_results[action][evaluator.f_measure]))
            print("{0} support : {1:d}".format(action, eval_results[action][evaluator.support]))
    else:
        print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'pre_train':
        pre_train(args)
    elif args.mode == 'fine_tune':
        fine_tune(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise RuntimeError('unknown mode')
