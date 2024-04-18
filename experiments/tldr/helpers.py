from typing import List, Dict
import random

import transformers


def format_example(
    example: List[Dict],
) -> Dict:
    """Formats example into a dictionary with keys for each constitution and response."""
    formatted_example = {}
    
    for i, constitution in enumerate(example): 
        
        prompt = f"{constitution['prompt']}"
        
        for j, response in enumerate(example): 
    
            response = response["response"]

            prompt_response = f"{prompt}{response}"
            formatted_example[f"prompt_c{i}_r{j}"] = prompt  
            formatted_example[f"response_c{i}_r{j}"] = prompt_response
            
    return formatted_example


def tokenize_func(
    example: Dict, 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize examples."""
    
    prompt_keys = [key for key in example.keys() if "prompt" in key]
    response_keys = [key for key in example.keys() if "response" in key]
    
    prompts = [example[key] for key in example.keys() if "prompt" in key]
    responses = [example[key] for key in example.keys() if "response" in key]
    
    tokenized_responses = [ # responses first for padding 
        tokenizer(
            response,
            add_special_tokens=True, 
            return_tensors="pt",
            padding=True,
        )
        for response in responses
    ]
        
    tokenized_prompts = [
        tokenizer(
            prompt,
            add_special_tokens=True,  
            return_tensors="pt",
            padding="max_length",
            max_length=tokenized_responses[i].input_ids.shape[1], # pad to the length of response
        )
        for i, prompt in enumerate(prompts)
    ]
    
    tokenized_example = {}
    
    for prompt_key, response_key, tokenized_prompt, tokenized_response in zip(
        prompt_keys, response_keys, tokenized_prompts, tokenized_responses
    ):
        for tokenized_key in ["input_ids", "attention_mask"]:
            tokenized_example[f"{prompt_key}_{tokenized_key}"] = tokenized_prompt[tokenized_key].squeeze(0)
            tokenized_example[f"{response_key}_{tokenized_key}"] = tokenized_response[tokenized_key].squeeze(0)
        
    return tokenized_example


def shuffle_principles(
    constitution: str,
) -> str:
    """Shuffle principles in a constitution."""
    principles = [
        principle.strip()[3:] 
        for _, principle in enumerate(constitution.split("\n"))
    ]
    
    random.shuffle(principles)

    principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
    shuffled_constitution = "\n".join(principles)
    
    return shuffled_constitution


def format_responses(
    responses: List[str],
) -> List[str]:
    """Check if generated responses follow desired format."""
    formatted_responses = ["", ""]
    try:
        formatted_responses[0] = f"The post {responses[0].strip().split('Human: ')[0].strip()}"
        formatted_responses[1] = f"The post {responses[1].strip().split('Human: ')[0].strip()}"
        if '###' in formatted_responses[0]:
            if "The post" in formatted_responses[0]:
                formatted_responses[0] = f"{formatted_responses[0].split('###')[0].strip()}"
            elif "The post" not in formatted_responses[0]:
                formatted_responses[0] = f"The post {formatted_responses[0].split('###')[0].strip()}"
        
        if '###' in formatted_responses[1]:
            if "The post" in formatted_responses[1]:
                formatted_responses[1] = f"{formatted_responses[1].split('###')[0].strip()}"
            elif "The post" not in formatted_responses[1]:
                formatted_responses[1] = f"The post {formatted_responses[1].split('###')[0].strip()}"
    except:
        print('Error in formatting responses. Skipping example.')
    
    return formatted_responses