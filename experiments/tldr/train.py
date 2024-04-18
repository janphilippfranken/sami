import os
import json
import random
import fire
import functools
import torch.nn as nn
import torch.multiprocessing as mp
import resource
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.mixtral.modeling_mistral import MistralDecoderLayer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import wandb
from sami.trainers.sami_trainer import SAMITrainer
from helpers import *

def get_policy_mixtral(blocks={MixtralDecoderLayer}): 
    """Wrapping policy setup mixtral."""
    return functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=blocks
    )
    
def get_policy_mistral(blocks={MistralDecoderLayer}):
    """Wrapping policy setup mistral."""
    return functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=blocks
    )

def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    port: int = 12355,
    backend: str = "nccl",
):
    print(rank, "initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def worker_main(rank: int, world_size: int, args: DictConfig, model):
    init_distributed(rank, world_size)
    if rank == 0:
        print("Initialized process group...")
        
        if args.wandb.log:
            wandb.init(project=args.wandb.project, name=args.wandb.name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # fsdp 
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    print(f"wrapping model {rank}...")
    
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    # check if model is mistral or mixtral
    is_mistral = 'mistral' in args.model.name.lower()
    
    model = FSDP(
        model,
        auto_wrap_policy=get_policy_mistral() if is_mistral else get_policy_mixtral(),
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False,
    )
    
    if is_mistral:
        check_fn = lambda submodule: isinstance(submodule, MistralDecoderLayer)
    else: 
        check_fn = lambda submodule: isinstance(submodule, MixtralDecoderLayer)
        
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    
    print(f"wrapped model {rank}...")

    # data 
    dataset_dict = json.load(open(os.path.join(args.data_path, args.data_file)))
    
    dataset_list = [
        format_example(example) for example in dataset_dict.values()
    ][:args.n_examples]
    
    if rank == 0:
        print(f"n examples: {len(dataset_list)}")

    train_dataset = [tokenize_func(example, tokenizer) for example in dataset_list]
    random.shuffle(train_dataset)
    
    if rank == 0:
        print(len(train_dataset))
        print(f"tokenized {len(train_dataset)} training examples...")
    
    # train
    trainer = SAMITrainer(
        model=model,
        reference_model=None, # if KL should be included, add reference model 
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=[],
        config=args,
        local_rank=rank,
        world_size=world_size,
    )
    
    trainer.train()


@hydra.main(version_base=None, config_path="conf", config_name="train_sami")
def main(args: DictConfig) -> None:
    
    torch.autograd.set_detect_anomaly(True)

    torch.cuda.manual_seed(args.training.seed)
    torch.manual_seed(args.training.seed)
    random.seed(args.training.seed)

    # model
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        torch_dtype=torch.bfloat16
    )
    
    # activation checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': True}
    )

    # spawn with resource limits
    world_size = torch.cuda.device_count()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
    mp.spawn(worker_main, nprocs=world_size, args=(world_size, args, model))


if __name__ == "__main__":
    fire.Fire(main())