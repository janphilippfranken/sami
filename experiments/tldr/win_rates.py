import json
import fire
import hydra
import random
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


from sami.models.openai_models.gpt4 import GPT4Agent
from sami.models.openai_models.azure import AsyncAzureChatLLM

from transformers import AutoTokenizer

from prompts import GPT4_WIN_RATE, SYSTEM_MESSAGE


def load_model_responses(filename):
    with open(filename, "r") as file:
        return json.load(file)
    

@hydra.main(version_base=None, config_path="conf", config_name="win_rates")
def main(args: DictConfig) -> None:
    
    model_baseline = load_model_responses(f"{args.model_dir}/{args.baseline}")
    model_test = load_model_responses(f"{args.model_dir}/{args.test}")
    
    print("BASELINE", args.baseline)
    print("TEST", args.test)
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        cache_dir="/scr/jphilipp/sami/pretrained_models/Mistral-7B-v0.1",
        model_max_length=2048,
    )
    
    win_rates = []
  
    llm = AsyncAzureChatLLM(
        azure_endpoint="https://philipp.openai.azure.com/",
        api_version="2023-05-15",
    )
    
    model = GPT4Agent(
        llm=llm,
        model="gpt-4",
        temperature=0.0,
        top_p=0.9,
        max_tokens=200,
        n=1,
    )

    constitutions = list(model_baseline['constitution'].values())
    questions = list(model_baseline['question'].values())

    random.seed(1)
    np.random.seed(1)
    prompts = []
    numbers = []
    lengths_base = []
    lengths_test = []
    
    for i, (constitution, question) in enumerate(zip(constitutions, questions)):
    
        if i >= 250: 
            continue
        
        principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
        random.shuffle(principles)
        principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
        constitution_shuffled = "\n".join(principles) 
       
        length_base = len(tokenizer.encode(model_baseline['response'][str(i)]))
        lengths_base.append(length_base)
       
        length_test = len(tokenizer.encode(model_test['response'][str(i)]))
        lengths_test.append(length_test)

        rand_number = np.random.randint(2)
        numbers.append(rand_number)
        
        if rand_number == 0:
        
            prompt = GPT4_WIN_RATE.format(
                constitution=constitution_shuffled,
                post=question,
                summary_a=model_baseline['response'][str(i)],
                summary_b=model_test['response'][str(i)],
            )
            prompts.append(prompt)
              
        elif rand_number == 1:
            prompt = GPT4_WIN_RATE.format(
                constitution=constitution_shuffled,
                post=question,
                summary_b=model_baseline['response'][str(i)],
                summary_a=model_test['response'][str(i)],
            )
               
            prompts.append(prompt)
        
    responses = model.batch_prompt(
        system_message=SYSTEM_MESSAGE,
        messages=prompts,
        win_rates=True,
    )
    
    formatted_responses = []

    for response in responses:
        try:
            formatted_responses.append(response[0].split('Final Response:')[1].strip())
        except:
            formatted_responses.append("C")

    for formatted_response, number, length_base, length_test in zip(formatted_responses, numbers, lengths_base, lengths_test):
       
        if number == 0:

            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates.append((0, length_base, length_test))
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates.append((0.5, length_base, length_test))
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates.append((1, length_base, length_test))
            else:
                print("ERROR")
                win_rates.append((0.5, length_base, length_test))
        
        elif number == 1:
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates.append((1, length_base, length_test))
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates.append((0.5, length_base, length_test))
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates.append((0, length_base, length_test))
            else:
                print("ERROR")
                win_rates.append((0.5, length_base, length_test))
                
    with open(f'{args.output_dir}/{args.win_rates_file_name}.json', 'w') as file:
        json.dump(win_rates, file, indent=4)
        
if __name__ == "__main__":
    fire.Fire(main())