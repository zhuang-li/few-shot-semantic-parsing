#!/bin/bash
parser='seq2seq_dong'
model_name=$(basename $1)

python exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --beam_size 5 \
    --parser ${parser} \
    --test_file datasets/jobs/query_split/shuffle_4_2/test.bin \
    --clip_grad_mode value \
    --decode_max_time_step 100
