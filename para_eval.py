import os
import json
import copy
from datetime import datetime
from tqdm import tqdm

from utils import math_equal, load_jsonl, save_jsonl
from parser import find_box, strip_string

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel

def extract_data(path, batch_size):
    data = []
    for p in load_jsonl(path):
        p['answer'] = strip_string(p['answer'])
        data.append(p)
    data = Dataset.from_list(data)
    return data

def copy_config(config):
    return copy.deepcopy(config)

def logging_inference(func):
    def wrapper(*args, **kwargs):
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print(f"program started at {current_time}")
        print("="*50)
        logs = copy_config(*args, **kwargs)
        logs['start_time'] = current_time
        prefix = os.getcwd() + "/temp/" + current_time
        os.makedirs(prefix, exist_ok=True)

        (accuracy, responses) = func(*args, **kwargs)

        print("="*50)
        print(f"The test accuracy on MATH500 is {accuracy}")
        if not logs['test']:
            save_jsonl(responses, prefix + "/responses.jsonl")
            logs['accuracy'] = accuracy
            with open(prefix + "/logs.json", "w", encoding='utf-8') as f:
                json.dump(logs, f, indent=4)
            print("Successfully run the full test, and relavent logs were saved.")

        return accuracy, responses
    return wrapper

@logging_inference
def run_cot_local_parallel(config):
    responses = []
    problem_cnt = 0
    solved_cnt = 0

    print(f"Now loading model: {config['model']}")
    print("="*50)

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], torch_dtype='auto'
    )
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=accelerator.device
    )
    
    data = extract_data(config['data_path'], config['batch_size'])
    for (n, d) in tqdm(enumerate(data), total=len(data)):
        messages = [
            {'role': 'system', 'content': config['sys_prompt']},
            {'role': 'user', 'content': d['problem']}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        cnt = str(pipe(text, max_new_tokens=4096, return_full_text=False))
        cnt = cnt[0]['generated_text']

        print(cnt)
        ans = strip_string(find_box(cnt))
        ground_truth = d['answer']
        correctness = math_equal(ground_truth, ans)
        problem_cnt += 1
        if correctness:
            solved_cnt += 1
        print(f"The extracted answer is: {ans}")
        print(f"And the ground truth answer is: {ground_truth}")
        print(f"And the math comparison gives: {correctness}")

        if not config['test']:
            d['pred_solution'] = cnt
            d['pred_answer'] = ans
            d['correctness'] = correctness
            responses.append(d)

        if config['test'] and n >= 5:
            break

    accuracy = float(solved_cnt) / float(problem_cnt)
    return accuracy, responses

@logging_inference
def run_lora_local_parallel(config):
    responses = []
    problem_cnt = 0
    solved_cnt = 0

    print(f"Now loading model: {config['model']}")
    print("="*50)

    accelerator = Accelerator()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **config['lora_config']
    )
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], torch_dtype='auto'
    )
    model = PeftModel.from_pretrained(model, config['adapter_path'])
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=accelerator.device
    )
    
    data = extract_data(config['data_path'], config['batch_size'])
    for (n, d) in tqdm(enumerate(data), total=len(data)):
        messages = [
            {'role': 'system', 'content': config['sys_prompt']},
            {'role': 'user', 'content': d['problem']}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        cnt = pipe(text, max_new_tokens=4096, return_full_text=False)
        cnt = cnt[0]['generated_text']

        print(cnt)
        ans = strip_string(find_box(cnt))
        ground_truth = d['answer']
        correctness = math_equal(ground_truth, ans)
        problem_cnt += 1
        if correctness:
            solved_cnt += 1
        print(f"The extracted answer is: {ans}")
        print(f"And the ground truth answer is: {ground_truth}")
        print(f"And the math comparison gives: {correctness}")

        if not config['test']:
            d['pred_solution'] = cnt
            d['pred_answer'] = ans
            d['correctness'] = correctness
            responses.append(d)

        if config['test'] and n >= 5:
            break

    accuracy = float(solved_cnt) / float(problem_cnt)
    return accuracy, responses
