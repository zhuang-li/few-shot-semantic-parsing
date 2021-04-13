#!/bin/bash
set -e

domain="housing"
data_dir="datasets/overnight/question_split/"
model_dir="saved_models/overnight/question_split/"
seed=${1:-0}
vocab=${data_dir}${domain}".vocab.freq2.bin"
train_file=${data_dir}${domain}"_train.bin"
dropout=0.5
dropout_i=0
hidden_size=256
embed_size=128
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=200
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=16
lr=0.001
alpha=0.95
ls=0.1
lstm='lstm'
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='seq2seq_topdown'
lang='overnight_lambda'
suffix='supvised'
attention='dot'
model_name=model.overn.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glo.ls${ls}.seed${seed}.cgm${clip_grad_mode}.p${parser}.suf${suffix}

python -u exp.py \
    --use_cuda \
    --seed ${seed} \
    --lang ${lang} \
    --mode train \
    --batch_size ${batch_size} \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --attention ${attention} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --parser ${parser} \
    --alpha ${alpha} \
    --decay_lr_every_epoch \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --clip_grad_mode ${clip_grad_mode} \
    --optimizer ${optimizer} \
    --lr_decay ${lr_decay} \
    --patience ${patience} \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 50 \
    --log_every 50 \
    --save_to ${model_dir}${model_name} \
    --dev_file ${data_dir}${domain}_test.bin \
    #--glove_embed_path embedding/glove/glove.6B.100d.txt \
    #2>logs/geo/lambda/question_split/${model_name}.log
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file ${data_dir}test.bin \
. scripts/overnight/question_split/test.sh ${model_dir}${model_name}.bin
#2>>logs/geo/lambda/question_split/${model_name}.log
