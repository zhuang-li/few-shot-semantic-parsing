#!/bin/bash
model_name=$(basename $1)
shuffle=$2
k_shot_num=$3
parser='seq2seq'
evaluator="smatch_evaluator"
data_dir="datasets/atis/atis_aug/shuffle_${shuffle}_shot_${k_shot_num}/"
python sup_exp.py \
    --use_cuda \
    --mode test \
    --lang atis_lambda \
    --load_model $1 \
    --beam_size 1 \
    --parser ${parser} \
    --relax_factor 10 \
    --evaluator ${evaluator} \
    --test_file ${data_dir}query.bin \
    --clip_grad_mode norm \
    --decode_max_time_step 100
