# SAMI: Self-Supervised Alignment with Mutual Information

### 🧐 What is this repo?
This repo includes a reference implementation of SAMI. 

## Setup 

`python train.py \
    training.beta=$beta \
    wandb.name="sami-lr-${lr}-iteration-${iteration}-opus" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/base" \
    data_file="base_mixtral_from_opus_principles.json" \
    n_examples=2000`