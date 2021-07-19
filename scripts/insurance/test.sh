#!/bin/bash
model_name=$(basename $1)
parser='seq2seq'
evaluator="default_evaluator"

data_dir_prefix="datasets/insurance/"
data_dir=${data_dir_prefix}
python sup_exp.py \
    --use_cuda \
    --mode test \
    --lang insurance_lambda \
    --load_model $1 \
    --beam_size 1 \
    --parser ${parser} \
    --evaluator ${evaluator} \
    --test_file ${data_dir}test.bin \
    --clip_grad_mode norm \
    --decode_max_time_step 100
