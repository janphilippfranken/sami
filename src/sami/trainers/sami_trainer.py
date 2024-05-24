from typing import (
    Dict, 
    List, 
    Optional, 
    Tuple,
    Sequence,
    
)

from dataclasses import dataclass

import os 
import tqdm

import wandb
from omegaconf import DictConfig

import torch
from torch.optim import AdamW, RMSprop
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig

import transformers
from transformers import get_scheduler, AutoModelForCausalLM

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)


def sami_loss(logprobs: torch.FloatTensor) -> torch.FloatTensor:
    """
    Args:
        logprobs: Shape (n_constitutions, n_responses)

    Returns:
        loss: Average cross-entropy loss.
    """
    logsumexp_row = torch.logsumexp(logprobs, dim=1, keepdim=True) # across responses
    logsumexp_col = torch.logsumexp(logprobs, dim=0, keepdim=True) # across constitutions
   
    logits_row = logprobs - logsumexp_row
    logits_col = logprobs - logsumexp_col

    labels_row = torch.arange(logits_row.shape[0], dtype=torch.long).to(logprobs.device) 
    labels_col = torch.arange(logits_col.shape[0], dtype=torch.long).to(logprobs.device)

    loss_row = F.cross_entropy(logits_row, labels_row, reduction="mean")
    loss_col = F.cross_entropy(logits_col.t(), labels_col, reduction="mean") # transpose col
    
    return (loss_col + loss_row) / 2


def _get_mask(
    attention_mask: torch.LongTensor,
    labels: torch.LongTensor,
    pad_token_id: int,
) -> torch.BoolTensor:
    """Returns a mask for the loss computation.
    
    Args:
        attention_mask: The attention mask of the prompt without final response. Shape: (batch_size, sequence_length).
        labels: The labels of the prompt including the final response. Shape: (batch_size, sequence_length).
        pad_token_id: The id of the padding token.
    
    Returns:
        mask: The mask for the loss computation. Shape: (batch_size, sequence_length).
    """
    prompt_mask = attention_mask.to(torch.bool) # mask prompt 
    response_mask = labels == pad_token_id      # mask padding
    
    # NOTE: this also masks the first eos token which is on purpose as during generation, we 'slice' the responses. 
    return torch.logical_or(
        prompt_mask,       # mask prompt
        torch.logical_and( # mask padding but not prompt
            torch.logical_not(prompt_mask),
            response_mask,
        ),
    ) 
    
    
def _get_mask_without_masking_eos(
    attention_mask: torch.LongTensor,
    labels: torch.LongTensor,
    model: torch.nn.Module,
    pad_token_id: int,
) -> torch.BoolTensor:
    """Returns a mask for the loss computation.
    
    Args:
        attention_mask: The attention mask of the prompt without final response. Shape: (batch_size, sequence_length).
        labels: The labels of the prompt including the final response. Shape: (batch_size, sequence_length).
        model: torch.nn.Module
        pad_token_id: The id of the padding token.
    
    Returns:
        mask: The mask for the loss computation. Shape: (batch_size, sequence_length).
    """
    prompt_mask = attention_mask.to(torch.bool) # mask prompt 
    
    # find first occurrence of pad which is eos token and do not mask that 
    is_pad = labels == pad_token_id
    cumsum_pad = torch.cumsum(is_pad, dim=1)
    eos_mask = cumsum_pad == 1
    eos_mask = eos_mask.to(model.device)
    response_mask = torch.logical_and(is_pad, ~eos_mask)
    response_mask = torch.logical_and(labels == pad_token_id, torch.logical_not(eos_mask))  # mask padding
     
    return torch.logical_or(
        prompt_mask,       # mask prompt
        torch.logical_and( # mask padding but not prompt
            torch.logical_not(prompt_mask),
            response_mask,
        ),
    ) 


def get_batch_logprobs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_idx: int = -100,
) -> torch.FloatTensor:
    """Computes the log probabilities of labels given logits.
    
    Args: 
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
    
    Returns:
        average_logprobs: The log probabilities of the labels. Shape: (batch_size, ).
    """
    assert logits.shape[:-1] == labels.shape, "Logits and labels must have the same shape."

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1]
    loss_mask = (labels == ignore_idx)
                                                                                                                                                                          
    per_token_logprobs = -F.cross_entropy( 
        input=logits.reshape(-1, logits.size(-1)),
        target=labels.reshape(-1), 
        reduction='none',
    ).reshape(labels.shape) 

    per_token_logprobs[loss_mask] = 0
    
    # average token logprobs
    average_token_logprobs = per_token_logprobs.sum(dim=-1) / torch.logical_not(loss_mask).sum(dim=1)
    
    return average_token_logprobs


def kl_divergence(
    policy_logits: torch.FloatTensor,
    ref_logits: torch.FloatTensor,
) -> torch.FloatTensor:
    """Computes the kl divergence between policy and reference model.
    
    Args:
        policy_logits: Shape: (batch_size, sequence_length, vocab_size)
        ref_logits: Shape: (batch_size, sequence_length, vocab_size)        
    """
    kl = (
        policy_logits.softmax(dim=-1) * (
            policy_logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))
        ).sum(dim=-1)

    return kl.mean()


def prepare_logits_labels(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: Dict[str, torch.Tensor],
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Prepares the logits and labels for the given prompts and responses. 
    
    Args:
        model: A torch.nn.Module model.
        tokenizer: A transformers.PreTrainedTokenizer.
        batch: A batch of tokenized examples. Each value should be a tensor of shape (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.
        
    Returns:
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
    """
    prompt_attention_mask = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "prompt" in key and 'attention_mask' in key 
        ],
        dim=0
    )
    
    response_attention_mask = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "response" in key and 'attention_mask' in key
        ],
        dim=0
    )
    
    responses = torch.cat(
        [
            batch[key] for key in batch.keys() 
            if "response" in key and 'attention_mask' not in key 
        ], 
        dim=0
    )

    labels = responses.clone()
        
    mask = _get_mask(
        attention_mask=prompt_attention_mask,
        labels=labels, 
        pad_token_id=tokenizer.pad_token_id,
    )

    labels[mask] = ignore_idx

    logits = model(
        input_ids=responses,
        attention_mask=response_attention_mask,
    ).logits

    return logits, labels


@dataclass
class SAMIDataCollator:
    """Collate examples."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            instances: A list of tokenized examples. Each dictionary should have keys for input_ids, attention_mask, and optionally labels.
        """
        collated_batch = {}
        unique_keys = instances[0].keys() if instances else []  # keys are expected to be shared across examples; only difference is actual content of prompts/responses

        max_length = max(
            (len(instance[key]) for instance in instances for key in unique_keys),
            default=0
        )

        for key in unique_keys:
            
            values = [
                torch.tensor(instance[key], dtype=torch.long) if not isinstance(instance[key], torch.Tensor)
                else instance[key] for instance in instances
            ]

            padding_value = self.tokenizer.pad_token_id if 'input_ids' in key else 0
            padded_values = [torch.nn.functional.pad(value, (0, max_length - value.size(0)), value=padding_value)
                                for value in values]
            
            collated_batch[key] = torch.stack(padded_values)

        return collated_batch
    
    
class SAMITrainer:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        reference_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: DictConfig,
        train_dataset: List[Dict],
        eval_dataset: List[Dict],
        local_rank: int,
        world_size: int,
    ):  
        """Intialize the samiTrainer.
        
        Args:
            model: transformers.PreTrainedModel.
            reference_model: transformers.PreTrainedModel.
            tokenizer: transformers.PreTrainedTokenizer.
            config: DictConfig.
            train_dataset: List of training examples. 
            eval_dataset: List of evaluation examples.
            config: Training configuration.  
            local_rank: the rank for distributed training
            world_size: num gpus 
        """
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        
        # data loaders 
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.training.train_batch_size, 
            collate_fn=SAMIDataCollator(tokenizer), 
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
            
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=config.training.eval_batch_size, 
            collate_fn=SAMIDataCollator(tokenizer),
            shuffle=False,
            sampler=DistributedSampler(eval_dataset),
        )

        # optimizer 
        if config.training.optimizer == "adamw": 
            self.optimizer = AdamW(model.parameters(), lr=config.training.lr)
        
        elif config.training.optimizer == "rmsprop":
            self.optimizer = RMSprop(model.parameters(), lr=config.training.lr)
 
        # scheduler 
        # self.scheduler = get_scheduler(
        #     name="linear", 
        #     optimizer=self.optimizer, 
        #     num_warmup_steps=config.training.num_warmup_steps, 
        #     num_training_steps=(
        #         len(self.train_dataloader) * self.config.training.n_epochs
        #         ) // config.training.gradient_accumulation_steps,
        # )

        # writing checkpoints
        self.checkpoint_dir = config.training.checkpoint_dir
        print("Loaded model on rank", self.local_rank)
        print("Loaded reference model on rank", self.local_rank)
        print(f"Writing checkpoints to {self.config.training.checkpoint_dir}.")
        dist.barrier()

    def save_model(
        self,
        epoch: int,
        state: Dict[str, torch.Tensor],
    ):
        """Merges checkpoint with HF model and writes to dir."""
        checkpoint_dir = os.path.join(self.config.training.checkpoint_dir, f"epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.config.training.save_option == "hf": # inefficient but stores HF format directly? 
            save_model = AutoModelForCausalLM.from_pretrained(
                **self.config.model.model_config,
            )
            save_model.load_state_dict(state)
            save_model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            print("Model saved, deleting model...")
            del save_model
            print("Deleted model...")
        
        elif self.config.training.save_option == "pt":
            torch.save(state, os.path.join(checkpoint_dir, "model.pt"))

    def save_checkpoint(self, epoch):
        """Save model, gathering from all processes."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            model_state_dict = self.model.state_dict()
            if self.local_rank == 0:
                self.save_model(
                    epoch=epoch,
                    state=model_state_dict,
                )
            del model_state_dict
        dist.barrier()

    def compute_metrics(
        self,
        model: transformers.PreTrainedModel,
        reference_model: transformers.PreTrainedModel,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Computes metrics for both policy and reference model."""
    
        # get logprobs for policy model 
        logits, labels = prepare_logits_labels(model, self.tokenizer, batch)
        
        batch_logprobs = get_batch_logprobs(logits, labels)
        
        # reshape to be n_constitutions * n_responses
        batch_logprobs = batch_logprobs.view(self.config.n_constitutions,  self.config.n_responses)
        
        # get logprobs for reference model 
        with torch.no_grad():
            if reference_model is not None:
                ref_logits, ref_labels = prepare_logits_labels(reference_model, self.tokenizer, batch)
            else:
                ref_logits, ref_labels = None, None

        # sami loss
        loss = sami_loss(logprobs=batch_logprobs)
                                              
        return loss, batch_logprobs, logits, ref_logits
    
    def _run_batch(
        self, 
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor], torch.FloatTensor]:
        """Run a batch."""    
        
        # compute policy and reference metrics
        loss, batch_logprobs, logits, ref_logits  = self.compute_metrics(self.model, self.reference_model, batch)
            
        if ref_logits is not None:
            kl_div = kl_divergence(
                policy_logits=logits,
                ref_logits=ref_logits,
            )

            # loss = loss + beta * kl
            adjusted_loss = loss + self.config.training.beta * kl_div
            
            return loss, adjusted_loss, batch_logprobs, kl_div
            
        else:
            
            return loss, loss, batch_logprobs, torch.tensor([0.0], device=self.local_rank)
        
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_raw_loss = 0.0
        total_kl_div = 0.0
        n_batches = 0

        # eval
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                raw_loss, loss,  _, kl_div = self._run_batch(batch)
                if torch.isnan(loss).any() or torch.isnan(raw_loss).any() or torch.isnan(kl_div).any():
                    if n_batches > 0:
                        average_loss = total_loss / n_batches
                        total_loss += average_loss
                        average_raw_loss = total_raw_loss / n_batches
                        total_raw_loss += average_raw_loss
                        average_kl_div = total_kl_div / n_batches
                        total_kl_div += average_kl_div
                    else:
                        total_loss += 0  
                        total_raw_loss += 0  
                        total_kl_div += 0 
                else:
                    total_loss += loss.item()
                    total_raw_loss += raw_loss.item()
                    total_kl_div += kl_div.item()
                n_batches += 1
                
        # logging 
        mean_loss = total_loss / n_batches
        mean_loss = torch.tensor([mean_loss], device=self.local_rank)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        reduced_loss = mean_loss / dist.get_world_size()
        
        mean_raw_loss = total_raw_loss / n_batches
        mean_raw_loss = torch.tensor([mean_raw_loss], device=self.local_rank)
        dist.all_reduce(mean_raw_loss, op=dist.ReduceOp.SUM)
        reduced_raw_loss = mean_raw_loss / dist.get_world_size()

        mean_kl_div = total_kl_div / n_batches
        mean_kl_div = torch.tensor([mean_kl_div], device=self.local_rank)
        dist.all_reduce(mean_kl_div, op=dist.ReduceOp.SUM)
        reduced_kl_div = mean_kl_div / dist.get_world_size()

        if self.local_rank == 0: 
            print(f"eval/loss: {reduced_loss.item()}")
            if self.config.wandb.log == True:
                wandb.log({"eval-loss/loss": reduced_loss.item()})
                wandb.log({"eval-loss/raw-loss": reduced_raw_loss.item()})
                wandb.log({"eval-loss/kld": reduced_kl_div.item()})

    def train(self):
        """Train the model."""
        if self.reference_model is not None:
            self.reference_model.eval()
        
        if self.config.training.evaluate_before_training:
            self.evaluate()
        
        for epoch in range(self.config.training.n_epochs):
            self.model.train()
            
            for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc=f"Running epoch: {epoch}"):
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                raw_loss, loss, batch_logprobs, kl_div = self._run_batch(batch)
   
                (loss / self.config.training.gradient_accumulation_steps).backward()
                
                # accumulate
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # logging
                loss_value = loss.item()
                loss_tensor = torch.tensor([loss_value], device=self.local_rank)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM) # maybe all_gather_if_needed instead? 
                reduced_loss = loss_tensor / dist.get_world_size()
                
                raw_loss_value = raw_loss.item() 
                raw_loss_tensor = torch.tensor([raw_loss_value], device=self.local_rank)
                dist.all_reduce(raw_loss_tensor, op=dist.ReduceOp.SUM)
                raw_reduced_loss = raw_loss_tensor / dist.get_world_size()

                dist.all_reduce(batch_logprobs, op=dist.ReduceOp.SUM)
                reduced_batch_logprobs = batch_logprobs / dist.get_world_size()
                
                dist.all_reduce(kl_div , op=dist.ReduceOp.SUM)
                reduced_kl_div = kl_div / dist.get_world_size()
                
                # hard-coded for 2 * 2 case atm. 
                p_c0_r0 = reduced_batch_logprobs[0, 0]
                p_c0_r1 = reduced_batch_logprobs[0, 1]
                p_c1_r0 = reduced_batch_logprobs[1, 0]
                p_c1_r1 = reduced_batch_logprobs[1, 1]

                accuracy_col = ((p_c0_r0 > p_c1_r0).int() + (p_c1_r1 > p_c0_r1).int()) / 2
                accuracy_row = ((p_c0_r0 > p_c0_r1).int() + (p_c1_r1 > p_c1_r0).int()) / 2

                if self.local_rank == 0:
                    print(f"Epoch {epoch}, Step {step}: train/loss = {reduced_loss.item()}, train/raw-loss = {raw_reduced_loss.item()}, train/logprobs = {reduced_batch_logprobs}, KL = {reduced_kl_div.item()}")
                    if self.config.wandb.log == True:
                        wandb.log({"train-loss/loss": reduced_loss.item()})
                        wandb.log({"train-loss/raw-loss": raw_reduced_loss.item()})
                        wandb.log({"train-loss/kld": reduced_kl_div.item()})
                        wandb.log({"train-probs/accuracy-col": accuracy_col.item()})
                        wandb.log({"train-probs/accuracy-row": accuracy_row.item()})
                        wandb.log({"train-probs/p_c0_r0": p_c0_r0.item()})
                        wandb.log({"train-probs/p_c0_r1": p_c0_r1.item()})
                        wandb.log({"train-probs/p_c1_r0": p_c1_r0.item()})
                        wandb.log({"train-probs/p_c1_r1": p_c1_r1.item()})
                    
                # evaluate and save after n steps have been made
                if self.config.training.save_after_n_steps:
                    if (step + 1) % self.config.training.save_after_n_steps == 0:
                        if self.config.training.evaluate:
                            self.evaluate()
                        self.save_checkpoint(round(step / len(self.train_dataloader), 2))
                                
        # evaluate at end of each epoch and save checkpoint 
        if self.config.training.evaluate:
            self.evaluate()
        self.save_checkpoint(epoch + 1)