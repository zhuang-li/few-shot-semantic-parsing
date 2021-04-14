#!/bin/bash

parser_name="seq2seq_few_shot"
evaluator='default_evaluator'
lang='geo_lambda'
model_name=$(basename $1)

shuffle=$2
k_shot_num=$3
data_dir="datasets/geo/freq_0/shuffle_${shuffle}_shot_${k_shot_num}/"
model_dir="saved_models/geo/freq_0/shuffle_${shuffle}_shot_${k_shot_num}/"

python dropout_few_shot_exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --lang ${lang} \
    --beam_size 1 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir}query.bin \
    --decode_max_time_step 50 \
