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
from model.seq2seq_final_few_shot import Seq2SeqModel
from model.seq2seq_few_shot import Seq2SeqModel
from model.seq2seq_few_shot_bin import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def fine_tune(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))

    print ("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg

    print("fine-tuning started ...")
    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        print ("=============================================================loading model===============================================================")
        model, src_vocab, vertex = parser_cls.load(model_path=args.load_model, use_cuda = args.use_cuda, loaded_vocab = vocab, args=args)

        print("load pre-trained word embedding (optional)")
        print (args.glove_embed_path)
        if args.glove_embed_path and (src_vocab is not None):
            #print (args.embed_size)
            print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
            glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
            glove_embedding.load_pre_train_to(model.src_embed, src_vocab)

    else:
        model = parser_cls(vocab, args)
    print ("setting model to training mode")
    model.few_shot_mode = 'fine_tune'
    model.train()

    evaluator = Registrable.by_name(args.evaluator)(args=args, vocab=vocab)
    if args.use_cuda and torch.cuda.is_available():
        model.cuda()
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    if args.optimizer == 'RMSprop':
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    examples = train_set.examples
    examples.sort(key=lambda e: -len(e.src_sent))
    model.init_action_embedding(examples, few_shot_mode='fine_tune')
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
                loss = torch.mean(loss)
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
                if att_probs:
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
            #model.init_action_embedding(train_set.examples)
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        is_better = False
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                #model.eval()
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
                #model.train()
        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch and epoch%args.valid_every_epoch == 0:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr



        if is_better:
            # model.init_action_embedding(train_set.examples)
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

def get_action_set(action_seq):
    action_set = set()
    for template_id, template_seq in enumerate(action_seq):
        for att_id, action in enumerate(template_seq):
            action_set.add(action)
    return action_set

def get_mask(action_set, vocab, use_cuda):
    label_mask = torch.zeros(1, len(vocab))
    if use_cuda:
        label_mask = label_mask.cuda()
    for action in list(action_set):
        # print (action)
        unmask_action_id = vocab[action]
        label_mask[0][0] = 1
        label_mask[0][1] = 1
        label_mask[0][unmask_action_id] = 1
    return label_mask

def pre_train(args):
    """Maximum Likelihood Estimation"""

    dir_path = os.path.dirname(args.train_file)

    temp_proto_loss_file = "pure_proto_loss.txt".format(args.att_reg)

    loss_file_path = os.path.join(dir_path, temp_proto_loss_file)

    loss_file = open(loss_file_path, 'w')

    # load in train/dev set
    train_set = EposideDataset.from_bin_file(args.train_file)

    vocab = pickle.load(open(args.vocab, 'rb'))

    print ("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    print("pre-training started ...")
    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        model = parser_cls.load(model_path=args.load_model, use_cuda = args.use_cuda)
    else:
        model = parser_cls(vocab, args)

    print ("setting model to training mode")

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

    print('begin training, %d training examples' % (len(train_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    action_set = get_action_set([e.tgt_actions for e in train_set.examples])
    # action mask
    action_mask = get_mask(action_set, model.action_vocab, args.use_cuda)

    model.train_action_set = action_set
    #print (action_set)
    model.action_mask = action_mask

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = report_ent_loss = report_att_reg_loss = 0.
    model.few_shot_mode = 'proto_train'
    while True:
        epoch += 1
        epoch_begin = time.time()

        for support_examples, query_examples in train_set.batch_iter(n_way=args.n_way, k_shot=args.k_shot, query_num=args.query_num, shuffle=True):
            #print (len(query_examples))
            train_iter += 1
            optimizer.zero_grad()
            #support_template = " ".join([str(action) for action in support_examples[0].tgt_actions])
            #query_template = " ".join([str(action) for action in query_examples[0].tgt_actions])
            #assert support_template == query_template, "support template must be the same with the query template {0} {1}".format(support_template, query_template)
            support_source = " ".join(support_examples[0].src_sent)
            query_source = " ".join(query_examples[0].src_sent)
            if args.few_shot_mode =='fine_tune':
                assert support_source == query_source, "support source must be the same with the query source {0} {1}".format(support_source, query_source)
            #else:
                #assert not (support_source == query_source), "support source must be the same with the query source {0} {1}".format(support_source, query_source)
            model.init_action_embedding(support_examples, few_shot_mode='proto_train')
            ret_val = model(query_examples)
            #print (model.action_proto[6])
            loss = ret_val[0]
            if loss.numel() > 1:
                #print ("loss size ", loss.size())
                loss_val = torch.sum(loss).data.item()
                loss = torch.mean(loss)
            elif loss.numel() == 1:
                #print ("1 loss size ", loss.size())
                loss_val = loss.item()
                loss = loss / len(query_examples)
            else:
                raise ValueError
            report_loss += loss_val
            report_examples += len(query_examples)

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    #print (len(att_probs))
                    sup_att_loss = -torch.stack(att_probs).mean()
                    sup_att_loss_val = sup_att_loss.data.item()
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            if args.att_reg:
                reg_loss = ret_val[2]
                if reg_loss:
                    att_reg_loss = torch.stack(reg_loss).mean()
                    att_reg_loss_val = att_reg_loss.data.item()
                    report_att_reg_loss += att_reg_loss_val

                    loss += att_reg_loss

            loss_ent = None
            if len(ret_val) > 2:
                loss_ent = ret_val[3]
                if loss_ent:
                    # print (len(loss_ent))
                    ent_loss = -torch.stack(loss_ent).mean()
                    ent_loss_val = -torch.stack(loss_ent).sum().data.item()
                    loss += ent_loss
                    report_ent_loss += ent_loss_val
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
                print(str(report_loss / report_examples), file=loss_file)

                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                if args.att_reg:
                    log_str += 'attention regularization loss=%.5f' % (
                                report_att_reg_loss / report_examples)
                    report_att_reg_loss = 0.

                if loss_ent:
                    log_str += ' entity loss=%.5f' % (report_ent_loss / report_examples)
                    report_ent_loss = 0

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            train_set.examples.sort(key=lambda e: -len(e.src_sent))
            model.init_action_embedding(train_set.examples)
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch and epoch%args.valid_every_epoch == 0:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        save_model_every_n_epoch = args.valid_every_epoch
        if epoch % save_model_every_n_epoch == 0:
            train_set.examples.sort(key=lambda e: -len(e.src_sent))
            print (len(train_set.examples))
            model.init_action_embedding(train_set.examples)

            model_file = args.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')


        if epoch == args.max_epoch:
            loss_file.close()
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)



def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    saved_args.use_cuda = args.use_cuda

    parser_cls = Registrable.by_name(args.parser)
    parser, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda, args=args)
    parser.few_shot_mode = 'fine_tune'
    parser.eval()
    vocab = parser.vocab
    evaluator = Registrable.by_name(args.evaluator)(args=args,vocab=vocab)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    if args.evaluator == 'action_evaluator':
        for action in eval_results:
            print ("{0} precision : {1:.4f}".format(action,eval_results[action][evaluator.precision]))
            print ("{0} recall : {1:.4f}".format(action,eval_results[action][evaluator.recall]))
            print ("{0} f measure : {1:.4f}".format(action,eval_results[action][evaluator.f_measure]))
            print ("{0} support : {1:d}".format(action,eval_results[action][evaluator.support]))
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
