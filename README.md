# SAMI: Self-Supervised Alignment with Mutual Information

## 🧐 What is this repo?
This repository includes a reference implementation of SAMI, Self-Supervised Alignment with Mutual Information. You can find further details [here](https://github.com/janphilippfranken/sami). 

Included in this repository is the data from the TL;DR experiments reported in our paper, located at `experiments/tldr/data`. It also contains responses sampled from each checkpoint, which were used to compute win rates in the paper.

## 🚀 Running the Code
### I have access to GPUs
#### 1. Generating Your Own Data (Optional)
After setting up your Conda environment and installing dependencies (`pip install -e .` should be sufficient), perform the following steps:
- Adjust the `generate.yaml` config to match your directories and the configurations you want to try. We have included `mistral-7b` and `claude-opus` as example configurations. The config file is available [here](https://github.com/janphilippfranken/sami/blob/main/experiments/tldr/conf/generate.yaml).
- Run `python generate.py`. By default, your data will be stored in `data/base`. This directory has been pre-populated with the data we generated if you prefer to fine-tune a model directly.

#### 2. Training
- Select a model configuration, such as `mistral-7b`, from [here](https://github.com/janphilippfranken/sami/blob/main/experiments/tldr/conf/model/mistral_7b_base.yaml). Update the `cache_dir` accordingly, for example, to a `/scr` path on your node.
- Adjust the `train_sami.yaml` config as needed. This config includes optional Weights & Biases logging. Make sure you are logged in if you set `log: true`. The config is available [here](https://github.com/janphilippfranken/sami/blob/main/experiments/tldr/conf/train_sami.yaml).
- Run training using an interactive job in Slurm with the following command:

  ```bash
  python train.py \
      training.beta=$beta \
      wandb.name="sami-lr-${lr}-iteration-${iteration}-opus" \
      training.checkpoint_dir="$checkpoint_dir" \
      training.lr=$lr \
      data_path="data/base" \
      data_file="base_mistral_from_opus_principles.json" \
      n_examples=2000
    ```

- Alternatively, adapt the example Slurm script to meet your computing needs and submit it using `sbatch`, or modify it to be a standard Bash script and submit using a tool like `tmux`. The script is available [here](https://github.com/janphilippfranken/sami/blob/main/experiments/tldr/example_scripts_slurm/train_typo_mixtral.sh).

#### 3. Evaluation
- Adjust the `evaluate.yaml` configuration and run `python evaluate.py`. The config file can be found [here](https://github.com/janphilippfranken/sami/blob/main/experiments/tldr/conf/evaluate.yaml). This will write responses into `experiments/tldr/results/responses`.
- Compute win rates by adjusting the `win_rates.yaml` configuration and running `python win_rates.py`. This script currently uses Azure, so if you lack access to GPT-4 via Azure, consider using a different pipeline for batch prompting or adapt the implementation in `azure.py` to use an AsyncOpenAI class. Both files can be found [here](https://github.com/janphilippfranken/sami/blob/main/src/sami/models/openai_models/azure.py).


#### I don't have access to GPUs
- You try to run training using [`mistral-tiny`](https://github.com/janphilippfranken/sami/blob/main/experiments/tldr/conf/model/mistral_tiny_base.yaml) which we ran locally on an `Apple M2 Pro` (2023 MacBook Pro with 16B memory) for debugging 


## 📖 Additional Resources
- Our repository is based on the [FSDP tutorial series](https://www.youtube.com/watch?v=8_k76AHu__s) and the [DDP tutorial series](https://www.youtube.com/watch?v=-K3bZYHYHEA&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj). 

## Citation
If you found this work useful, please cite:
```bibtex
@article{ref,
    title={ref}, 
    author={ref},
    year={2024},
    eprint={ref},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}




