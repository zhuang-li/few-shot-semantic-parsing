#!/bin/bash
data_dir="datasets/atis/question_split/"
parser_name="seq2seq"
evaluator="default_evaluator"
model_name=$(basename $1)

python exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --beam_size 1 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir}test.bin \
    --decode_max_time_step 100 \
    --att_reg 0
