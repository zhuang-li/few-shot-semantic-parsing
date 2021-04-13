#!/bin/bash
set -e

seed=${1:-0}
vocab="datasets/jobs/query_split/shuffle_4_2/vocab.freq2.bin"
train_file="datasets/jobs/query_split/shuffle_4_2/train.bin"
dropout=0.4
hidden_size=150
lr_decay=0.985
lr_decay_after_epoch=5
max_epoch=95
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=128
lr=0.01
lstm='lstm'
optimizer='RMSprop'
clip_grad_mode='value'
parser='seq2seq_dong'
model_name=model.geo.sup.${lstm}.hid${hidden_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mod${clip_grad_mode}.pars${parser}

python -u exp.py \
    --use_cuda \
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --optimizer ${optimizer} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --clip_grad_mode ${clip_grad_mode} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --uniform_init 0.08 \
    --beam_size ${beam_size} \
    --decode_max_time_step 100 \
    --log_every 50 \
    --parser ${parser} \
    --save_to saved_models/jobs/${model_name}
    #2>logs/geo/question_split/question_split/${model_name}.log

. scripts/jobs/test.sh saved_models/jobs/${model_name}.bin 2>>logs/jobs/${model_name}.log
