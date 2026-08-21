#!/bin/bash
#SBATCH  -J pretrain
#SBATCH  -t 10-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia

#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:h100:1


##SBATCH  --constraint=80g
##SBATCH -q shamout
#SBATCH -q cair
##SBATCH -q nvidia-xxl






export TOKENIZER_PATH="./resources/vocab.json"
# export DATA_PATH="./data/meds_normalized_arrow"

# export DATA_IDX_PATH="./resources/pretrain_index.parquet"

export WANDB_API_KEY="59b6438e0496b3089f91abef35d31dae69b6c009"
export LOG_DIR="./models/pretraining/"

export VERSION="15_maskprob_12.5overlap"
export PRETRAIN_MODE="simclr"


export DATA_PATH="./data/desc_gen_dataset/"
export DATA_IDX_PATH="./resources/downstream_index.parquet"



eval "$(conda shell.bash hook)"
conda activate ehr-ragp

set -x



# export BACKBONE="bert"
# export BASELINE="None"
# torchrun --nproc_per_node=2 pretrain.py \
#     --learning-rate 1e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 64 \
#     --chunk-length 512 \
#     --overlap 64 


export BACKBONE="genhpf_simclr"
export BASELINE="genhpf"
torchrun --nproc_per_node=1 pretrain.py \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --max-epochs 100 \
    --batch-size 64 \
    --chunk-length 510 \
    --overlap 0 



# export BACKBONE="roberta"
# export BASELINE="transformer"
# torchrun --nproc_per_node=1 pretrain.py \
#     --learning-rate 1e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 64 \
#     --chunk-length 512 \
#     --overlap 64



# export BACKBONE="longformer"
# export BASELINE="transformer"
# torchrun --nproc_per_node=1 pretrain.py \
#     --learning-rate 1e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 16 \
#     --chunk-length 1024 \
#     --overlap 128



# export BACKBONE="big_bird"
# export BASELINE="transformer"
# torchrun --nproc_per_node=1 pretrain.py \
#     --learning-rate 1e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 16 \
#     --chunk-length 1024 \
#     --overlap 128



# export BACKBONE="modernbert"
# export BASELINE="transformer"
# module load gcc/12.2.0
# torchrun --nproc_per_node=1 pretrain.py \
#     --learning-rate 1e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 16 \
#     --chunk-length 1024 \
#     --overlap 128


# export BACKBONE="roformer"
# export BASELINE="transformer"
# torchrun --nproc_per_node=1 pretrain.py \
#     --learning-rate 1e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 8 \
#     --chunk-length 1024 \
#     --overlap 128


# export BACKBONE="mamba"
# torchrun --nproc_per_node=4 pretrain.py \
#     --learning-rate  \
#     --weight-decay 1e-5 \
#     --max-epochs 100 \
#     --batch-size 8 \
#     --chunk-length 2048 \
#     --overlap 0