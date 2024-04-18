#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:8                       
#SBATCH --mem=512GB                       
#SBATCH --cpus-per-task=64               
#SBATCH --time=12:00:00                    
#SBATCH --output=train_sami.out         
#SBATCH --error=train_sami.err           

# cond env
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate sami

cd ~/research_projects/sami/experiments/tldr


beta=0.0    # 0 -> no KL
lr=5e-7     # learning rate 
iteration=1 # curr iteration
checkpoint_dir="/scr/jphilipp/sami/trained_models/Mixtral-8x7b-v.01/checkpoints-sumarization-opus/sami-${lr}-iteration-${iteration}-opus"

python train.py \
    training.beta=$beta \
    wandb.name="sami-lr-${lr}-iteration-${iteration}-opus" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/base" \
    data_file="base_mixtral_from_opus_principles.json" \
    n_examples=2000 