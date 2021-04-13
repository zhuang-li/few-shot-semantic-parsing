#!/bin/bash
#PBS -P dz21
#PBS -q gpuvolta
#PBS -l walltime=12:00:00
#PBS -l mem=32GB
#PBS -l jobfs=32GB
#PBS -l ngpus=1
#PBS -l ncpus=12

source /scratch/dz21/zl1166/venv/bin/activate
module load cuda/10.1

domains=('basketball' 'blocks' 'calendar' 'housing' 'publications' 'recipes' 'restaurants' 'socialnetwork')
data_dir="/scratch/dz21/zl1166/cross_template_semantic_parsing/datasets/overnight/continual_split/"
model_dir="/scratch/dz21/zl1166/cross_template_semantic_parsing/saved_models/overnight/continual_split/"
decode_dir="/scratch/dz21/zl1166/cross_template_semantic_parsing/datasets/overnight/decode_results/"
seed=0
dropout=0.5
dropout_i=0
hidden_size=256
num_exemplars_per_class=5
num_exemplars_per_task=100
num_exemplars_ratio=0.01
num_known_domains=2
embed_size=128
replay_dropout=1
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=100
max_epoch=100
patience=1000   # disable patience since we don't have dev set
beam_size=1
augment=0
batch_size=64
lr=0.001
alpha=0.95
ls=0.1
num_memory_buffer=1000
sample_method=$sample
lstm='lstm'
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='gss_emr'
lang='overnight_lambda'
suffix='sup'
attention='dot'
model_name=model.overn.${lstm}.exemplars_ratio${num_exemplars_ratio}.sample_method${sample_method}.num_memory_buffer${num_memory_buffer}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glo.seed${seed}.p${parser}.suf${suffix}

python3 -u /scratch/dz21/zl1166/cross_template_semantic_parsing/life_exp.py \
    --use_cuda \
    --seed ${seed} \
    --lang ${lang} \
    --mode train \
    --batch_size ${batch_size} \
    --train_file ${data_dir} \
    --test_file ${data_dir} \
    --vocab ${data_dir} \
    --lstm ${lstm} \
    --attention ${attention} \
    --augment ${augment} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --num_exemplars_per_task ${num_exemplars_per_task} \
    --num_exemplars_ratio ${num_exemplars_ratio} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --max_epoch ${max_epoch} \
    --num_memory_buffer ${num_memory_buffer} \
    --sample_method ${sample_method} \
    --lr ${lr} \
    --replay_dropout ${replay_dropout} \
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
    --domains ${domains[@]} \
    --num_exemplars_per_class ${num_exemplars_per_class} \
    --num_known_domains ${num_known_domains} \
    --save_decode_to ${decode_dir} > ${model_name}.out
    #--dev_file ${data_dir} \

