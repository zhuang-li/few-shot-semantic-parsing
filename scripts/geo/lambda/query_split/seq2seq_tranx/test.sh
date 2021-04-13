#!/bin/bash
model_name=$(basename $1)
shuffle=$2
k_shot_num=$3
evaluator='smatch_evaluator' # smatch_evaluator default_evaluator
parser='seq2seq'
data_dir="datasets/geo/supervised/shuffle_${shuffle}_shot_${k_shot_num}/"
python sup_exp.py \
    --use_cuda \
    --mode test \
    --lang geo_lambda \
    --load_model $1 \
    --beam_size 1 \
    --parser ${parser} \
    --relax_factor 10 \
    --evaluator ${evaluator} \
    --test_file ${data_dir}test.bin \
    --clip_grad_mode norm \
    --decode_max_time_step 100
