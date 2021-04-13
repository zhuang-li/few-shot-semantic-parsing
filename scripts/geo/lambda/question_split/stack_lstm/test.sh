#!/bin/bash
data_dir="datasets/geo/lambda/query_split/few_shot_split/"
parser_name="seq2seq_complete"
evaluator="default_evaluator"
lang='geo_lambda'
model_name=$(basename $1)

python exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --lang ${lang} \
    --beam_size 5 \
    --relax_factor 10 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir}test.bin \
    --decode_max_time_step 100
