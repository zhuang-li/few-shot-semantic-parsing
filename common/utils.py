# coding=utf-8
import argparse

import os
import logging
import time


def config_logger(log_prefix):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/logs/' + log_prefix + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + rq + '.log'
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger

class cached_property(object):

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--use_cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--lang', choices=['geo_prolog', 'geo_lambda', 'atis_lambda', "job_prolog"], default='geo_lambda',
                            help='language to parse. Deprecated, use --transition_system and --parser instead')
    arg_parser.add_argument('--mode', choices=['pre_train', 'fine_tune', 'test'], default='train', help='Run mode')
    arg_parser.add_argument('--few_shot_mode', choices=['pre_train', 'fine_tune', 'test'], default='pre_train', help='Few shot running mode')

    arg_parser.add_argument('--metric', choices=['prototype', 'relation', 'dot', 'matching'], default='dot', help='Metrics')
    arg_parser.add_argument('--k_shot', default=1, type=int, help='number of eposide shots')
    arg_parser.add_argument('--n_way', default=5, type=int, help='number of eposide types')
    arg_parser.add_argument('--query_num', default=1, type=int, help='number of query instances in one eposide')
    arg_parser.add_argument('--task_num', default=4, type=int, help='number of tasks in one batch')

    #### Modularized configuration ####
    arg_parser.add_argument('--parser', type=str, default='seq2seq', required=False, help='name of parser class to load')
    arg_parser.add_argument('--evaluator', type=str, default='default_evaluator', required=False, help='name of evaluator class to use')

    #### Model configuration ####
    arg_parser.add_argument('--lstm', choices=['lstm','parent'], default='lstm', help='Type of LSTM used, currently only standard LSTM cell is supported')

    arg_parser.add_argument('--attention', choices=['dot','general', 'concat', 'bahdanau'], default='dot', help='Type of LSTM used, currently only standard LSTM cell is supported')

    arg_parser.add_argument('--sup_attention', action='store_true', default=False, help='Use supervised attention')

    arg_parser.add_argument('--att_reg',  default=1., type=float, help='Use attention regularization')

    arg_parser.add_argument('--reg',  default=1., type=float, help='Use attention regularization')

    arg_parser.add_argument('--train_mask', action='store_true', default=False, help='Use mask during training')
    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')

    arg_parser.add_argument('--use_children_lstm_encode', default=False, action='store_true',
                            help='Use children lstm encode')
    arg_parser.add_argument('--use_input_lstm_encode', default=False, action='store_true',
                            help='Use input lstm encode')
    arg_parser.add_argument('--use_att', default=True, action='store_true',
                            help='Use att encoding in the next time step')

    arg_parser.add_argument('--use_coverage', default=False, action='store_true',
                            help='Use coverage attention in the next time step')
    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--glove_embed_path', default=None, type=str, help='Path to pretrained Glove mebedding')

    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--support_file', type=str, help='path to the support source file')
    arg_parser.add_argument('--query_file', type=str, help='path to the query source file')


    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--train_iter', default=5, type=int, help='Train iteration size')
    arg_parser.add_argument('--dropout_i', default=0., type=float, help='Input Dropout rate')
    arg_parser.add_argument('--dropout', default=0., type=float, help='Dropout rate')
    arg_parser.add_argument('--word_dropout', default=0., type=float, help='Word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='Word dropout rate on decoder')
    arg_parser.add_argument('--label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing when predicting labels')

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int, help='Perform validation every x epoch')
    arg_parser.add_argument('--sup_proto_turnover', default=2, type=int, help='supervised proto turn over rate')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')
    arg_parser.add_argument('--forward_pass', default=10, type=int, help='forward pass n iterations in maml')
    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true', help='Save all intermediate checkpoints')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='If specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true', help='Use glorot initialization')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument('--clip_grad_mode', choices=['value', 'norm'], required=True, help='clip gradients type')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    arg_parser.add_argument('--step_size', default=0.001, type=float, help='Inner Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--proto_dropout', default=0.5, type=float,
                            help='drop out for the proto probability')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--lr_decay_every_n_epoch', default=1, type=int, help='Decay learning rate every n epoch')
    arg_parser.add_argument('--decay_lr_every_epoch', action='store_true', default=False, help='force to decay learning rate after each epoch')
    arg_parser.add_argument('--alpha', default=0.95, type=float, help='alpha rate for the rmsprop')
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False, help='Whether to reset optimizer when loading the best checkpoint')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=True,
                            help='Only evaluate the top prediction in validation')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')

    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')
    # validate grammar

    arg_parser.add_argument('--share_parameter', default=False, action='store_true',
                            help='Use shared parameter or not')

    arg_parser.add_argument('--hyp_embed', default=True, action='store_true',
                            help='Use hybrid shared parameter or not')

    arg_parser.add_argument('--proto_embed', default=False, action='store_true',
                            help='Use proto embedding or not')



    return arg_parser


def update_args(args, arg_parser):
    for action in arg_parser._actions:
        if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreTrueAction) \
                or isinstance(action, argparse._StoreFalseAction):
            if not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)
