#!/bin/bash
data_dir="datasets/geo/lambda/query_split/few_shot_split_random_5_predi/align_threshold_0.5/"
parser_name="seq2seq_complete"
evaluator="default_evaluator"
lang='geo_lambda'
model_name=$(basename $1)

python maml_exp.py \
    --use_cuda \
    --few_shot_mode test \
    --load_model $1 \
    --lang ${lang} \
    --beam_size 5 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --evaluator ${evaluator} \
    --test_file ${data_dir}query.bin \
    --decode_max_time_step 50
