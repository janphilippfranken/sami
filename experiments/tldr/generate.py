import json
import fire
import random
import hydra
import numpy as np
import os
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset

from sami.models.vllm_models.inference_model import VLLMInferenceModel

from helpers import *
from prompts import *


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(args: DictConfig) -> None:
    print(args.model_config)
    print(args.start_example, args.max_example)
    print(isinstance(args.start_example, int), isinstance(args.max_example, int), isinstance(args.batch_size, int))
    
    random.seed(42)
        
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )

    # data 
    dataset = load_dataset(
        args.dataset.path,
        'comparisons',
        cache_dir=args.dataset.cache_dir,
        split=args.dataset.split,
    )['info']
    
    filtered_dataset = []
    filtered_ids = []
    
    for example in dataset:
        if example['id'] not in filtered_ids:
            filtered_dataset.append(example)
            filtered_ids.append(example['id'])
    filtered_dataset = filtered_dataset[args.start_example:args.max_example]

    constitutions = [
        json.load(open(f"{args.constitution_dir}/{constitution}")) 
        for constitution in os.listdir(args.constitution_dir)
    ]
    
    # data dict to store 
    train_data = {} 
    
    # batch containers
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_start_index = 0  

    for i, example in enumerate(tqdm(filtered_dataset, desc="Processing examples")):
        constitution = random.choice(constitutions)
        constitution_positive = constitution['positive']
        constitution_negative = constitution['negative']

        question = f"POST: {example['post']}"
        batch_questions.append(question)
        
        # generation prompts
        generation_prompts = [
            PROMPT_GENERATION.format(constitution=constitution, question=question)
            for constitution in [constitution_positive, constitution_negative]
        ]
        batch_prompts.extend(generation_prompts)

        # shuffle constitutions for training prompts
        positive_constitution_shuffled = shuffle_principles(constitution_positive)
        negative_constitution_shuffled = shuffle_principles(constitution_negative)
        batch_train_constitutions.append([positive_constitution_shuffled, negative_constitution_shuffled])

        if (i + 1) % args.batch_size == 0 or i == len(filtered_dataset) - 1:
            batch_responses = model.batch_prompt(
                prompts=batch_prompts,
                **args.generation_config,
            )
            
            for j in range(0, len(batch_responses), 2):
                responses = batch_responses[j:j+2]
                responses_positive, responses_negative = responses[0], responses[1]
                formatted_positive, formatted_negative = "", ""
    
                formatted_responses = format_responses([responses_positive, responses_negative])
 
                # filtering for unwanted terms like "["
                if all(substring not in formatted_responses[0] for substring in args.filter):
                    formatted_positive = formatted_responses[0]
                if all(substring not in formatted_responses[1] for substring in args.filter):
                    formatted_negative = formatted_responses[1]
            
                if formatted_positive and formatted_negative:
                    if formatted_positive[-1] == "." and formatted_positive != formatted_negative:
                        question_index = int(j / 2)
                        
                        data =[
                            {
                                "prompt": PROMPT_TRAINING.format(constitution=batch_train_constitutions[question_index][k], question=batch_questions[question_index]),
                                "response": response,
                                "example_id": batch_start_index + question_index,
                            }
                            for k, response in zip(
                                range(2),

                                [formatted_positive, formatted_negative]
                            )
                        ]

                        train_data[batch_start_index + question_index] = data
                    
                    else:
                        print(f"Skipping example {batch_start_index + int(j / 2)}")
                else:
                    print(f"Skipping example {batch_start_index + int(j / 2)}")
        
            # reset for the next batch
            batch_prompts = []
            batch_train_constitutions = []
            batch_questions = []
            batch_start_index += len(batch_responses) // 2 
     
            with open(f"{args.output_dir}/{args.file_name}.json", "w") as file:
                json.dump(train_data, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())