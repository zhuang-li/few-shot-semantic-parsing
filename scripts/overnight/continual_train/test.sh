#!/bin/bash
data_dir="datasets/overnight/continual_split/"
parser_name="seq2seq_topdown"
evaluator="default_evaluator"
lang='overnight_lambda'
model_name=$(basename $1)
domains=("housing" "calendar")

python continual_exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --lang ${lang} \
    --beam_size 1 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir} \
    --new_domains ${domains[@]} \
    --decode_max_time_step 100
