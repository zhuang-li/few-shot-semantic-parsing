#!/bin/bash
data_dir="datasets/geo/lambda/question_split/recursive_lstm/"
model_name=$(basename $1)

python exp.py \
    --use_cuda \
    --mode test \
    --load_model $1 \
    --beam_size 5 \
    --relax_factor 10 \
    --parser seq2seq_c_t \
    --clip_grad_mode norm \
    --test_file ${data_dir}test.bin \
    --decode_max_time_step 100
