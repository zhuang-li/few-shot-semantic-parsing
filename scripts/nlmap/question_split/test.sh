#!/bin/bash
data_dir="datasets/nlmap/question_split/stack_lstm/"
parser_name="seq2seq_c_t_action"
model_name=$(basename $1)
lang='nlmap'
python exp.py \
    --use_cuda \
    --mode test \
    --lang ${lang} \
    --load_model $1 \
    --beam_size 5 \
    --parser ${parser_name} \
    --clip_grad_mode norm \
    --test_file ${data_dir}test.bin \
    --decode_max_time_step 100
