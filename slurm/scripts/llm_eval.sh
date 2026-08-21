#!/bin/bash
#SBATCH  -J zllm
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:h200:1

##SBATCH  --constraint=80g
##SBATCH -q cair
#SBATCH -q shamout
##SBATCH  -q nvidia-xxl




export HF_TOKEN=$(cat ~/.hf_token)
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

export HF_HOME="../huggingface_cache"
export TRANSFORMERS_CACHE="../huggingface_cache"
export HUGGINGFACE_HUB_CACHE="../huggingface_cache"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export CC="/share/apps/NYUAD5/gcc/9.2.0/bin/gcc"
export CXX="/share/apps/NYUAD5/gcc/9.2.0/bin/g++"


# Qwen/Qwen2.5-7B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3

# google/medgemma-1.5-4b-it
# BioMistral/BioMistral-7B
python llm_eval.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-path ./data/llm-dataset \
  --data-idx-path ./resources/downstream_index.parquet \
  --split held_out

