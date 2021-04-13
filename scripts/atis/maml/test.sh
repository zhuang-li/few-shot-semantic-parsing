#!/bin/bash

parser_name="seq2seq_complete"
evaluator="default_evaluator"
lang='atis_lambda'
model_name=$(basename $1)

shuffle=$2
k_shot_num=$3
data_dir="datasets/atis/query_split/few_shot_split_random_5_predi/shuffle_${shuffle}_shot_${k_shot_num}/"
model_dir="saved_models/atis/query_split/few_shot_split_random_5_predi/shuffle_${shuffle}_shot_${k_shot_num}/"

python maml_exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --lang ${lang} \
    --beam_size 1 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir}query.bin \
    --decode_max_time_step 50
