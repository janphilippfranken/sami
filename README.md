# SAMI: Self-Supervised Alignment with Mutual Information

## Introduction

This repository contains a reference implementation of SAMI (Self-Supervised Alignment with Mutual Information), a novel approach for aligning language models using self-supervised learning. The repository includes the source code, experimental data, and detailed instructions for running the code and reproducing the results reported in the accompanying paper.

## Repository Structure

- `experiments/tldr/data`: Contains the data from the TL;DR experiments reported in the paper, along with responses sampled from each checkpoint used to compute win rates.
- `experiments/tldr/conf`: Contains various configuration files for data generation, training, and evaluation.
- `experiments/tldr/example_scripts_slurm`: Contains example Slurm scripts for running training jobs.
- `experiments/tldr/results/responses`: Contains the responses generated during the evaluation phase.
- `src/sami/models/openai_models/azure.py`: Contains the implementation of the Azure-based batch prompting pipeline for computing win rates.

## Running the Code

### Prerequisites

- Set up a Conda environment and install the required dependencies by running `pip install -e .`.
- Ensure you have access to GPUs for training and evaluation.

### Data Generation (Optional)

1. Adjust the `generate.yaml` config file to match your directories and desired configurations. Example configurations for `mistral-7b` and `claude-opus` are provided.
2. Run `python generate.py` to generate your own data. By default, the generated data will be stored in `data/base`. Note that this directory is pre-populated with the data used in the paper if you prefer to fine-tune a model directly.

### Training

1. Select a model configuration (e.g., `mistral-7b`) from the `experiments/tldr/conf/model` directory and update the `cache_dir` accordingly.
2. Adjust the `train_sami.yaml` config as needed, including optional Weights & Biases logging. Ensure you are logged in if you set `log: true`.
3. Run training using an interactive job in Slurm with the provided command, or adapt the example Slurm script to meet your computing needs and submit it using `sbatch`. Alternatively, modify the script to be a standard Bash script and submit using a tool like `tmux`.

### Evaluation

1. Adjust the `evaluate.yaml` configuration and run `python evaluate.py`. This will write the generated responses into `experiments/tldr/results/responses`.
2. Compute win rates by adjusting the `win_rates.yaml` configuration and running `python win_rates.py`. Note that this script currently uses Azure, so if you lack access to GPT-4 via Azure, consider using a different pipeline for batch prompting or adapt the implementation in `azure.py` to use an AsyncOpenAI class.

### Running without GPUs

If you don't have access to GPUs, you can attempt to run training using the `mistral-tiny` configuration, which was tested locally on an Apple M2 Pro (2023 MacBook Pro with 16B memory) for debugging purposes.

## Additional Resources

This repository is based on the [FSDP tutorial series](https://www.youtube.com/watch?v=8_k76AHu__s) and the [DDP tutorial series](https://www.youtube.com/watch?v=-K3bZYHYHEA&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj).

## Citation

If you find this work useful, please cite:

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




