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

from model.nn_utils import to_input_variable
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
from model.locked_dropout import ProtoDropout
from model.seq2seq_bi_action import Seq2SeqModel
from model.seq2seq_template import Seq2SeqModel
from model.seq2seq_final_few_shot import Seq2SeqModel
from model.seq2seq_few_shot import Seq2SeqModel
from model.seq2seq_few_shot_plain import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def get_model_loss(model, batch_examples, args, report_loss, report_examples, report_sup_att_loss, report_att_reg_loss, report_ent_loss):
    ret_val = model(batch_examples)
    loss = ret_val[0]

    if loss.numel() > 1:
        # print ("loss size ", loss.size())
        loss_val = torch.sum(loss).data.item()
        loss = torch.mean(loss)
    elif loss.numel() == 1:
        # print ("1 loss size ", loss.size())
        loss_val = loss.item()
        loss = loss / len(batch_examples)
    else:
        raise ValueError
    report_loss += loss_val
    report_examples += len(batch_examples)

    if args.sup_attention:
        att_probs = ret_val[1]
        if att_probs:
            # print (len(att_probs))
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

    loss_ent = ret_val[3]
    if loss_ent:
        ent_loss = -torch.stack(loss_ent).mean()
        ent_loss_val = ent_loss.data.item()
        report_ent_loss += ent_loss_val
        loss += ent_loss

    return loss, report_loss, report_examples, report_sup_att_loss, report_att_reg_loss, report_ent_loss

def fine_tune(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    # remove examples
    """
    remove_examples = []
    remove_id = 4
    for e_id, e in enumerate(train_set.examples):
        if e_id == remove_id:
            continue
        else:
            remove_examples.append(e)
    print(len(remove_examples))
    train_set = Dataset(remove_examples)
    """
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
        model, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda = args.use_cuda, loaded_vocab=vocab,args=args)
        print("load pre-trained word embedding (optional)")
        print (args.glove_embed_path)
        if args.glove_embed_path and (src_vocab is not None):
            #print (args.embed_size)
            print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
            glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
            glove_embedding.load_pre_train_to(model.src_embed, src_vocab)

        if args.init_vertex and args.glove_embed_path and vertex_vocab is not None:
            print ("================================================= init vertex ============================================")
            print (vertex_vocab)
            word_ids, non_init_vertex = init_vertex_embedding_with_glove(model.ast_embed, vertex_vocab, vocab.predicate_tokens, args)
            print ("non init vertex")
            print (non_init_vertex)
    else:
        model = parser_cls(vocab, args)
    print ("setting model to fintuning mode")
    model.few_shot_mode = 'fine_tune'
    model.train()

    #init_vertex_embedding_with_glove(model.ast_embed, model.vertex_vocab.token2id, vocab.predicate_tokens, args)

    evaluator = Registrable.by_name(args.evaluator)(args=args)
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
    report_loss = report_examples = report_sup_att_loss = report_att_reg_loss = report_ent_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    train_set.examples.sort(key=lambda e: -len(e.src_sent))
    update_action_freq(model.action_freq, [e.tgt_actions for e in train_set.examples])
    print("init action embedding")
    #print (args.metric)
    if args.metric == 'matching':
        model.proto_train_action_set = dict()
        #print ("=================================")
        #print (model.proto_train_action_set)
    model.init_action_embedding(train_set.examples, few_shot_mode='fine_tune')
    #if args.num_exemplars_per_task:
        #subselect_examples = train_set.random_sample_batch_iter(args.num_exemplars_per_task)
        #replay_dataset = Dataset(subselect_examples)
    #print (model.proto_train_action_set.keys())
    print("finish init action embedding")
    while True:
        if not args.max_epoch == 0:
            epoch += 1
            epoch_begin = time.time()

            for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
                train_iter += 1
                optimizer.zero_grad()

                loss, report_loss, report_examples, report_sup_att_loss, report_att_reg_loss, report_ent_loss = \
                    get_model_loss(model, batch_examples, args, report_loss, report_examples, report_sup_att_loss, report_att_reg_loss, report_ent_loss)
                """
                if args.num_exemplars_per_task:
                    #print (args.num_exemplars_per_task)
                    #print (len(batch_examples))
                    replay_examples = replay_dataset.random_sample_batch_iter(len(batch_examples))
                    replay_examples.sort(key=lambda e: -len(e.src_sent))
                    replay_loss, report_loss, report_examples, report_sup_att_loss, report_att_reg_loss, report_ent_loss = \
                        get_model_loss(model, replay_examples, args, report_loss, report_examples, report_sup_att_loss,
                                       report_att_reg_loss, report_ent_loss)
                    loss += replay_loss
                """
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
                    if args.att_reg:
                        log_str += ' attention regularization loss=%.5f' % (report_att_reg_loss / report_examples)
                        report_att_reg_loss = 0.
                    log_str += ' entity attention loss=%.5f' % (report_ent_loss / report_examples)
                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = report_ent_loss = 0.

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
                model.eval()
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
                model.train()
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

def update_action_freq(action_freq, action_seq):

    for template_id, template_seq in enumerate(action_seq):
        template_seq_list = list(set(template_seq))
        for att_id, action in enumerate(template_seq_list):
            if action in action_freq:
                action_freq[action] = action_freq[action] + 1
            else:
                action_freq[action] = 1

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

def train_iter_func(iter_obj, model, proto_drop, optimizer, args, train_mode = 'proto_train' , src_ids = [], vertex_ids = []):
    if train_mode == 'supervised_train':
        batch_examples = next(iter_obj)
        #print (len(batch_examples))
        model.few_shot_mode = 'supervised_train'
        ret_val = model(batch_examples)
        length_examples = len(batch_examples)
        #print (batch_examples[0].src_sent)
    elif train_mode == 'proto_train':
        support_examples, query_examples = next(iter_obj)
        #print ("support",len(support_examples))
        #print ("query",len(query_examples))
        model.few_shot_mode = 'proto_train'
        #print (model.proto_mask)
        assert model.proto_mask == None, "there should be no mask before init the embedding"
        assert model.proto_train_action_set == None, "there should be no proto train action set before init the embedding"
        model.proto_train_action_set = dict()
        action_set = model.init_action_embedding(support_examples, few_shot_mode='proto_train')
        proto_mask = get_mask(action_set, model.action_vocab, model.use_cuda)
        model.proto_mask = proto_mask

        #
        vocab_embedding = model.action_embedding(model.new_long_tensor([i for i in range(len(model.action_vocab))]))
        vocab_embedding = (vocab_embedding.t() * model.action_mask).t()

        proto_mask[0][0] = 0
        proto_mask[0][1] = 0
        #print ("before -------------")
        #for i in range(model.action_proto.size(0)):
            #print (model.action_proto[i])

        model.action_proto = model.action_proto.clone() + (vocab_embedding.t() * (1.0 - proto_mask)).t()


        drop_proto, mask = proto_drop(model.action_proto)
        #print (drop_proto)

        """
        proto_train_action_set = {}

        for idx, bool_value in enumerate(list(mask[0])):
            if bool_value > 0:
                proto_train_action_set.add(model.action_vocab.id2token[idx])

        model.proto_train_action_set = proto_train_action_set
        """
        #print (mask)
        #print (mask)
        #print (drop_proto.size())


        #print (vocab_embedding.size())
        model.action_proto = drop_proto + (vocab_embedding.t() * (1.0 - mask)).t()
        #for i in range(model.action_proto.size(0)):
            #print (model.action_proto[i])
        #print (model.action_embedding.weight.size())
        #print (model.action_embedding.weight[0].requires_grad)
        ret_val = model(query_examples)
        length_examples = len(query_examples)
        model.proto_mask = None
        model.proto_train_action_set = None

    # print (len(query_examples))
    optimizer.zero_grad()


    # print (model.action_proto[6])
    loss = ret_val[0]
    if loss.numel() > 1:
        # print ("loss size ", loss.size())
        loss_val = torch.sum(loss).data.item()
        loss = torch.mean(loss)
    elif loss.numel() == 1:
        # print ("1 loss size ", loss.size())
        loss_val = loss.item()
        loss = loss / length_examples
    else:
        raise ValueError
    sup_att_loss_val = 0
    if args.sup_attention:
        att_probs = ret_val[1]
        if att_probs:
            # print (len(att_probs))
            sup_att_stack_loss = torch.stack(att_probs)
            sup_att_loss = -sup_att_stack_loss.mean()
            sup_att_loss_val = -sup_att_stack_loss.sum().data.item()

            loss += sup_att_loss

    att_reg_loss_val = 0

    if args.att_reg:
        reg_loss = ret_val[2]
        if reg_loss:
            # print (len(att_probs))
            reg_stack_loss = torch.stack(reg_loss)
            att_reg_loss = reg_stack_loss.mean()
            att_reg_loss_val = reg_stack_loss.sum().data.item()

            loss += att_reg_loss

    ent_loss_val = 0
    loss_ent = ret_val[3]
    if loss_ent:
        #print (len(loss_ent))

        ent_stack_loss = torch.stack(loss_ent)

        ent_loss = -ent_stack_loss.mean()
        ent_loss_val = -ent_stack_loss.sum().data.item()
        loss += ent_loss

    loss.backward()

    #print(model.att_reg_linear.weight.grad)

    if args.embed_fixed:
        if src_ids:
            #print (model.new_long_tensor(src_ids))
            model.src_embed.weight.grad[model.new_long_tensor(src_ids)] = 0
        if args.init_vertex:
            if vertex_ids:
                #print (model.new_long_tensor(vertex_ids))
                #print (vertex_ids)
                model.ast_embed.weight.grad[model.new_long_tensor(vertex_ids)] = 0
                #print (model.ast_embed.weight.grad[model.new_long_tensor(vertex_ids)].size())


    # clip gradient
    if args.clip_grad > 0.:
        if args.clip_grad_mode == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        elif args.clip_grad_mode == 'value':
            grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

    optimizer.step()

    return loss_val, length_examples, sup_att_loss_val, att_reg_loss_val, ent_loss_val

def init_vertex_embedding_with_glove(vertex_embed_layer, vertex_vocab, predicate_token_vocab, args):
    predicate_embed = torch.nn.Embedding(len(predicate_token_vocab), args.action_embed_size)
    torch.nn.init.xavier_normal_(predicate_embed.weight.data)
    if args.use_cuda:
        predicate_embed = predicate_embed.cuda()
    # print (src_embed.weight[12].data)
    #print (args.glove_embed_path)
    predicate_glove_embedding = GloveHelper(args.glove_embed_path, args.action_embed_size)
    predicate_glove_embedding.load_to(predicate_embed, predicate_token_vocab)
    #print (vertex_vocab)
    #print (predicate_token_vocab.lemma_token2id)
    non_init_vertex = []
    word_ids = []
    for vertex, id in vertex_vocab.items():
        if len(vertex.prototype_tokens) == 0:
            non_init_vertex.append(vertex)
            continue
        vertex_var = to_input_variable([vertex.prototype_tokens], predicate_token_vocab, args.use_cuda,
                                   append_boundary_sym=False)
        vertex_embeddings = predicate_embed(vertex_var)
        vertex_embedding = vertex_embeddings.mean(dim=0).squeeze()
        #print (vertex_embedding)
        vertex_embed_layer.weight.data[id].copy_(vertex_embedding)
        word_ids.append(id)


    del predicate_embed
    torch.cuda.empty_cache()
    return word_ids, non_init_vertex

def pre_train(args):
    """Maximum Likelihood Estimation"""

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])

    dir_path = os.path.dirname(args.train_file)

    temp_sup_loss_file = "reg_{}_sup_loss.txt".format(args.att_reg)
    temp_proto_loss_file = "reg_{}_proto_loss.txt".format(args.att_reg)

    sup_loss_file_path = os.path.join(dir_path, temp_sup_loss_file)

    proto_loss_file_path = os.path.join(dir_path, temp_proto_loss_file)

    sup_loss_file = open(sup_loss_file_path, 'w')

    proto_loss_file = open(proto_loss_file_path, 'w')

    # load in train/dev set
    normal_train_set = Dataset.from_bin_file(args.train_file)

    eposide_train_set = EposideDataset.from_bin_file(args.train_file)


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
    evaluator = Registrable.by_name(args.evaluator)(args=args)
    model.train()

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

    src_ids = []
    vertex_ids = []
    print("load pre-trained word embedding (optional)")
    if args.glove_embed_path:
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
        src_ids = glove_embedding.load_to(model.src_embed, vocab.source)

        if args.init_vertex:
            print ("init vertex embedding with glove")
            vertex_ids, non_init_vertex = init_vertex_embedding_with_glove(model.ast_embed, model.vertex_vocab.token2id, vocab.predicate_tokens, args)
            print ("non init vertex")
            print (non_init_vertex)


    print('begin training, %d training examples' % (len(normal_train_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = report_att_reg_loss = report_ent_loss = 0.

    proto_report_loss = proto_report_examples = proto_report_sup_att_loss = proto_report_att_reg_loss =  proto_report_ent_loss = 0.

    proto_drop = ProtoDropout(args.proto_dropout)
    epoch_flag = False
    epoch_begin = time.time()

    sup_iter_obj = normal_train_set.batch_iter(batch_size=args.batch_size, shuffle=True)

    proto_iter_obj = eposide_train_set.batch_iter(n_way=args.n_way, k_shot=args.k_shot, query_num=args.query_num, shuffle=True)
    # action set
    action_set = get_action_set([e.tgt_actions for e in normal_train_set.examples])

    update_action_freq(model.action_freq, [e.tgt_actions for e in normal_train_set.examples])

    # action mask
    action_mask = get_mask(action_set, model.action_vocab, args.use_cuda)

    model.train_action_set = action_set
    #print (action_set)
    model.action_mask = action_mask
    #print (action_mask)
    while True:
        if train_iter%args.sup_proto_turnover == 0:
            try:
                loss_val, length_examples, sup_att_loss_val, att_reg_loss_val, ent_loss_val = train_iter_func(sup_iter_obj, model, proto_drop, optimizer, args, train_mode = 'supervised_train', src_ids = src_ids, vertex_ids = vertex_ids)

                report_loss += loss_val
                report_examples += length_examples

                if args.sup_attention:
                    report_sup_att_loss += sup_att_loss_val

                if args.att_reg:
                    report_att_reg_loss += att_reg_loss_val

                report_ent_loss +=ent_loss_val

                if train_iter % args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                    print(str(report_loss / report_examples), file=sup_loss_file)
                    if args.sup_attention:
                        log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                        report_sup_att_loss = 0.
                    if args.att_reg:
                        log_str += ' attention regularization loss=%.5f' % (report_att_reg_loss / report_examples)
                        report_att_reg_loss = 0.

                    log_str += ' entity attention loss=%.5f' % (report_ent_loss / report_examples)
                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = report_ent_loss = 0.

            except StopIteration:
                sup_iter_obj = normal_train_set.batch_iter(batch_size=args.batch_size, shuffle=True)
                epoch += 1
                epoch_flag = True
                print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
                epoch_begin = time.time()
                continue
        else:
            try:
                loss_val, length_examples, sup_att_loss_val, att_reg_loss_val, ent_loss_val = train_iter_func(proto_iter_obj, model, proto_drop, optimizer, args, train_mode = 'proto_train', src_ids = src_ids, vertex_ids = vertex_ids)

                proto_report_loss += loss_val
                proto_report_examples += length_examples

                if args.sup_attention:
                    proto_report_sup_att_loss += sup_att_loss_val

                if args.att_reg:
                    proto_report_att_reg_loss += att_reg_loss_val

                proto_report_ent_loss += ent_loss_val

                if train_iter % args.log_every == 1:
                    log_str = '[Iter %d] proto encoder loss=%.5f' % (train_iter, proto_report_loss / proto_report_examples)
                    print(str(proto_report_loss / proto_report_examples), file=proto_loss_file)
                    if args.sup_attention:
                        log_str += ' proto supervised attention loss=%.5f' % (proto_report_sup_att_loss / proto_report_examples)
                        proto_report_sup_att_loss = 0.

                    if args.att_reg:
                        log_str += ' proto attention regularization loss=%.5f' % (proto_report_att_reg_loss / proto_report_examples)
                        proto_report_att_reg_loss = 0.

                    log_str += ' entity attention loss=%.5f' % (
                                proto_report_ent_loss / proto_report_examples)

                    print(log_str, file=sys.stderr)
                    proto_report_loss = proto_report_examples = proto_report_ent_loss = 0.

            except StopIteration:
                proto_iter_obj = eposide_train_set.batch_iter(n_way=args.n_way, k_shot=args.k_shot,
                                                              query_num=args.query_num, shuffle=True)
                continue



        train_iter += 1
        if epoch_flag:

            if args.dev_file:
                # model.eval()
                normal_train_set.examples.sort(key=lambda e: -len(e.src_sent))
                model.few_shot_mode = 'proto_train'
                assert model.few_shot_mode == 'proto_train'
                model.init_action_embedding(normal_train_set.examples,few_shot_mode='proto_train')
                #model.few_shot_mode == 'fine_tune'
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


            if args.save_all_models:
                normal_train_set.examples.sort(key=lambda e: -len(e.src_sent))
                model.few_shot_mode = 'proto_train'
                assert model.few_shot_mode == 'proto_train'
                if args.metric == 'matching':
                    model.proto_train_action_set = dict()
                model.init_action_embedding(normal_train_set.examples,few_shot_mode='proto_train')
                if args.metric == 'matching':
                    model.proto_train_action_set = None

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
                normal_train_set.examples.sort(key=lambda e: -len(e.src_sent))
                model.few_shot_mode = 'proto_train'
                assert model.few_shot_mode == 'proto_train'
                if args.metric == 'matching':
                    model.proto_train_action_set = dict()
                model.init_action_embedding(normal_train_set.examples,few_shot_mode='proto_train')
                if args.metric == 'matching':
                    model.proto_train_action_set = None
                model_file = args.save_to + '.bin'
                print('save the current model ..', file=sys.stderr)
                print('save model to [%s]' % model_file, file=sys.stderr)
                #print (len(model.train_action_set))
                model.save(model_file)
                # also save the optimizers' state
                torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')


            if epoch == args.max_epoch:
                sup_loss_file.close()
                proto_loss_file.close()
                print('reached max epoch, stop!', file=sys.stderr)
                exit(0)
            epoch_flag = False


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model
    print (args.lang)
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    saved_args.use_cuda = args.use_cuda

    parser_cls = Registrable.by_name(args.parser)
    parser, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda, args=args)
    parser.few_shot_mode = 'fine_tune'
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