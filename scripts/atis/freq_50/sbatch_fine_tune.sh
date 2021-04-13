#!/bin/bash
#SBATCH --job-name=cross_template
# To set a project account for credit charging,
#SBATCH --account=fz25


# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Request for GPU,

#SBATCH --gres=gpu:1

#SBATCH --partition=m3c

# Memory usage (MB)
#SBATCH --mem=16000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=0-15:00:00


# To receive an email when job completes or fails
#SBATCH --mail-user=zhuang.li@monash.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=cross_template_atis_proto-%j.out

# Set the file for error log (stderr)
#SBATCH --error=cross_template_atis_proto-%j.err


# Command to run a gpu job
# For example:
module load cuda
nvidia-smi
deviceQuery
source ~/fz25/cross_template_env/bin/activate

set -e

seed=0
pre_model_name=$(basename $1)
shuffle=$2
k_shot_num=$3
suffix=$4
data_dir="datasets/atis/query_split/few_shot_split_random_5_predishuffle_${shuffle}_shot_${k_shot_num}/"
model_dir="saved_models/atis/query_split/few_shot_split_random_5_predishuffle_${shuffle}_shot_${k_shot_num}/"

vocab=${data_dir}"support_vocab.bin"
support_file=${data_dir}"support.bin"

dropout=0.5
dropout_i=0
hidden_size=256
embed_size=200
att_reg=1
action_embed_size=128
lr_decay=0.985
proto_dropout=0.5
lr_decay_after_epoch=20
max_epoch=30
patience=1000   # disable patience since we don't have dev set
batch_size=64
beam_size=1
n_way=30
k_shot=1
query_num=3
lr=0.0025
alpha=0.95
ls=0.1
relax_factor=10
lstm='lstm'
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser='seq2seq_few_shot'
#pre_model_name=model.geo.pre_train.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.relax_factor${relax_factor}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}
fine_model_name=model.atis.fine_tune.${lstm}.hid${hidden_size}.shuffle.${shuffle}.shot.${k_shot}.embed${embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${support_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.parser${parser}.suffix${suffix}


python -u dropout_few_shot_exp.py \
    --use_cuda \
    --seed ${seed} \
    --mode fine_tune \
    --lang atis_lambda \
    --att_reg ${att_reg} \
    --valid_every_epoch 1 \
    --n_way ${n_way} \
    --k_shot ${k_shot} \
    --query_num ${query_num} \
    --train_file ${support_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --proto_dropout ${proto_dropout} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --alpha ${alpha} \
    --parser ${parser} \
    --clip_grad_mode ${clip_grad_mode} \
    --optimizer ${optimizer} \
    --patience ${patience} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --glorot_init \
    --decode_max_time_step 50 \
    --batch_size ${batch_size} \
    --beam_size ${beam_size} \
    --log_every 10 \
    --load_model $1 \
    --dev_file ${data_dir}query.bin \
    --save_to ${model_dir}${fine_model_name} \
    --glove_embed_path embedding/glove/glove.6B.200d.txt \

. scripts/atis/dropout_proto_few_shot/test.sh ${model_dir}${fine_model_name}.bin ${shuffle} ${k_shot_num} ${suffix}

