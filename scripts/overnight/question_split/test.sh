#!/bin/bash
data_dir="datasets/overnight/question_split/"
parser_name="seq2seq_topdown"
evaluator="default_evaluator"
lang='overnight_lambda'
domain="housing"
model_name=$(basename $1)

python exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --lang ${lang} \
    --beam_size 1 \
    --relax_factor 10 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir}${domain}_test.bin \
    --decode_max_time_step 100
