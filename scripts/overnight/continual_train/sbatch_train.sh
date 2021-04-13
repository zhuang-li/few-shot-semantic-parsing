#!/bin/bash
#SBATCH --job-name=cross_template
# To set a project account for credit charging,
#SBATCH --account=fz25


# Request CPU resource for a serial job
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4

# Request for GPU,
#SBATCH --gres=gpu:1

#SBATCH --partition=m3g

# Memory usage (MB)
#SBATCH --mem=32000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=08:00:00


# To receive an email when job completes or fails
#SBATCH --mail-user=zhuang.li@monash.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=continual_learning-%j.out

# Set the file for error log (stderr)
#SBATCH --error=continual_learning-%j.err


# Command to run a gpu job
# For example:
module load cuda
module load cmake
nvidia-smi
deviceQuery
source ~/fz25/cross_template_env/bin/activate
#module load pytorch

set -e

domains=('basketball' 'blocks' 'calendar' 'housing' 'publications' 'recipes' 'restaurants' 'socialnetwork')
data_dir="datasets/overnight/continual_split/"
model_dir="saved_models/overnight/continual_split/"
decode_dir="datasets/overnight/decode_results/"
seed=$1 # seed for permutations
subselect=0
ewc=0
mode="train"
p_seed=0
dropout=0
dropout_i=0
hidden_size=256
num_exemplars_per_class=5
num_exemplars_per_task=50
num_exemplars_ratio=0.01
num_known_domains=1
embed_size=128
replay_dropout=1
action_embed_size=128
lr_decay=0.985
lr_decay_after_epoch=100
max_epoch=10
patience=1000   # disable patience since we don't have dev set
beam_size=1
rebalance=0
augment=0
batch_size=64
lr=0.0025
alpha=0.95
ls=0.1
reg=0
num_memory_buffer=500
sample_method=$2
lstm='lstm'
optimizer='Adam'
clip_grad_mode='norm'
hierarchy_label='default'
parser=$3
lang='overnight_lambda'
suffix='sup'
attention='dot'
evaluator='denotation_evaluator'
model_name=model.overn.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glo.ls${ls}.seed${seed}.cgm${clip_grad_mode}.p${parser}.sa_me${sample_method}

python -u life_exp.py \
    --use_cuda \
    --seed ${seed} \
    --p_seed ${p_seed} \
    --lang ${lang} \
    --mode ${mode} \
    --batch_size ${batch_size} \
    --train_file ${data_dir} \
    --test_file ${data_dir} \
    --ewc ${ewc} \
    --vocab ${data_dir} \
    --evaluator ${evaluator} \
    --lstm ${lstm} \
    --reg ${reg} \
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
    --rebalance ${rebalance} \
    --sample_method ${sample_method} \
    --subselect ${subselect} \
    --num_memory_buffer ${num_memory_buffer} \
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
    --save_decode_to ${decode_dir}
    #--dev_file ${data_dir} \

