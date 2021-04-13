#!/bin/bash
# SBATCH --job-name=cross_template
# To set a project account for credit charging,
# SBATCH --account=zlii0182


# Request CPU resource for a serial job
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=4

# Request for GPU,

# SBATCH --gres=gpu:1


# Memory usage (MB)
# SBATCH --mem=8000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
# SBATCH --time=0-01:00:00


# To receive an email when job completes or fails
# SBATCH --mail-user=zhuang.li@monash.au
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL


# Set the file for output (stdout)
# SBATCH --output=cross_template-%j.out

# Set the file for error log (stderr)
# SBATCH --error=cross_template-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a gpu job
# For example:
module load cuda/10.0
nvidia-smi
deviceQuery
source ~/fz25/cross_template_env/bin/activate


set -e

seed=${1:-0}
vocab="datasets/geo/lambda/question_split/vocab.freq2.bin"
train_file="datasets/geo/lambda/question_split/train.bin"
dropout=0.5
dropout_i=0.5
hidden_size=150
embed_size=150
action_embed_size=150
lr_decay=0.985
lr_decay_after_epoch=10
max_epoch=250
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=64
lr=0.005
alpha=0.95
ls=0.1
relax_factor=10
lstm='lstm'
optimizer='RMSprop'
clip_grad_mode='norm'
hierarchy_label='default'
parser='seq2seq_c_t'
model_name=model.geo.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.drop${dropout}.dropout_i${dropout_i}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.relax_factor${relax_factor}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}

while getopts "d:r:h:e:a:l:f:o:k:p:m:" opt; do
  case $opt in
    d)
      dropout=$OPTARG
      echo "dropout is $dropout"
      ;;
    r)
      dropout_i=$OPTARG
      echo "dropout_i is $dropout_i"
      ;;
    h)
      hidden_size=$OPTARG
      echo "hidden_size is $hidden_size"
      ;;
    e)
      embed_size=$OPTARG
      echo "embed_size is $embed_size"
      ;;
    a)
      action_embed_size=$action_embed_size
      echo "action_embed_size is $action_embed_size"
      ;;
    l)
      lr=$OPTARG
      echo "lr is $lr"
      ;;
     f)
      alpha=$OPTARG
      echo "alpha is $alpha"
      ;;
     o)
      optimizer=$OPTARG
      echo "optimizer is $optimizer"
      ;;
     k)
      lr_decay=$OPTARG
      echo "lr_decay is $lr_decay"
      ;;
     p)
      lr_decay_after_epoch=$OPTARG
      echo "lr_decay_after_epoch is $lr_decay_after_epoch"
      ;;
     m)
      max_epoch=$OPTARG
      echo "max_epoch is $max_epoch"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

python -u exp.py \
    --use_cuda \
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --train_file ${train_file} \
    --hierarchy_label ${hierarchy_label} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --label_smoothing ${ls} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --dropout_i ${dropout_i} \
    --patience ${patience} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --alpha ${alpha} \
    --parser ${parser} \
    --clip_grad_mode ${clip_grad_mode} \
    --optimizer ${optimizer} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --uniform_init 0.08 \
    --beam_size ${beam_size} \
    --relax_factor ${relax_factor}\
    --decode_max_time_step 100 \
    --log_every 50 \
    --dev_file datasets/geo/lambda/question_split/test.bin \
    --save_to saved_models/geo/lambda/question_split/${model_name}
    #2>logs/geo/lambda/question_split/${model_name}.log
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--dev_file datasets/geo/lambda/question_split/test.bin \
. scripts/geo/lambda/question_split/test.sh saved_models/geo/lambda/question_split/${model_name}.bin
#2>>logs/geo/lambda/question_split/${model_name}.log

