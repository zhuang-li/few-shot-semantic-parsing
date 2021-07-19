from __future__ import print_function
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import time

import six
import argparse

from flask import Flask, jsonify, render_template, request
import json


from evaluation import evaluate
from common.registerable import Registrable
from components.dataset import Example
from model.seq2seq import Seq2SeqModel
from components.evaluator import DefaultEvaluator

app = Flask(__name__)
parsers = dict()

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--cuda', action='store_true', default=True, help='Use gpu')
    arg_parser.add_argument('--config_file', type=str, required=True,
                            help='Config file that specifies model to load, see online doc for an example')
    arg_parser.add_argument('--port', type=int, required=False, default=8081)

    return arg_parser


def decode(examples, model, decode_max_time_step, beam_size, verbose=False):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    for example in examples:
        hyps = model.beam_search(example, decode_max_time_step, beam_size=beam_size)
        decode_results.extend(hyps)
    if was_training: model.train()

    return decode_results


@app.route('/parse/<lang>/', methods=['GET'])
def parse(lang):
    utterance = request.args['q']

    parser = parsers[lang]

    if six.PY2:
        utterance = utterance.encode('utf-8', 'ignore')

    decode_results = decode([Example(utterance.split(' '), [], [], [], idx=0, meta=None)], parser, 100, 5, verbose=False)

    responses = dict()
    responses['hypotheses'] = []

    for hyp_id, hyp in enumerate(decode_results):
        print('------------------ Hypothesis %d ------------------' % hyp_id)
        print(hyp.to_logic_form)

        hyp_entry = dict(id=hyp_id + 1,
                         value=hyp.to_logic_form,
                         score=hyp.score)

        responses['hypotheses'].append(hyp_entry)

    return jsonify(responses)


if __name__ == '__main__':
    args = init_arg_parser().parse_args()
    config_dict = json.load(open(args.config_file))

    for lang, config in config_dict.items():
        parser_id = config['parser_type']
        parser_cls = Registrable.by_name(parser_id)
        parser = parser_cls.load(model_path=config['model_path'], use_cuda=args.cuda, args=args)
        parser.eval()
        vocab = parser.vocab

        parsers[lang] = parser

    app.run(host='0.0.0.0', port=args.port, debug=True)
