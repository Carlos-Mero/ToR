import os
import copy
import json
from datetime import datetime
from utils import math_equal, load_jsonl, save_jsonl, find_boxed
from openai import OpenAI
from tqdm import tqdm

def extract_data(path):
    data = []
    for p in load_jsonl(path):
        data.append(p)
    return data

def run_cot(config):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"program started at {current_time}")
    print("="*50)
    if not config['test']:
        prefix = os.getcwd() + "/temp/" + current_time
        os.makedirs(prefix, exist_ok=True)
        logs = copy.deepcopy(config)
        logs['start_time'] = current_time
        responses = []
    problem_cnt = 0
    solved_cnt = 0
    
    client = OpenAI().chat.completions
    data = extract_data(config['data_path'])
    for (n, d) in tqdm(enumerate(data), total=len(data)):
        messages = [
            {'role': 'system', 'content': config['sys_prompt']},
            {'role': 'user', 'content': d['problem']}
        ]
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
        )

        cnt = completion.choices[0].message.content
        print(cnt)
        ans = find_boxed(cnt)
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

    # save the logs of the single run
    print("="*50)
    accuracy = float(solved_cnt) / float(problem_cnt)
    print(f"The test accuracy on MATH500 is {accuracy}")
    if not config['test']:
        save_jsonl(responses, prefix + "/responses.jsonl")
        logs['problems_count'] = problem_cnt
        logs["solved"] = solved_cnt
        logs['accuracy'] = accuracy
        with open(prefix + "/logs.json", "w", encoding='utf-8') as f:
            json.dump(logs, f, indent=4)
        print("Successfully run the full test, and relavent logs were saved.")

def run_with_guidance(config):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"program started at {current_time}")
    print("="*50)
    if not config['test']:
        prefix = os.getcwd() + "/temp/" + current_time
        os.makedirs(prefix, exist_ok=True)
        logs = copy.deepcopy(config)
        logs['start_time'] = current_time
        responses = []
    problem_cnt = 0
    solved_cnt = 0
    
    client = OpenAI().chat.completions
    data = extract_data(config['data_path'])
    for (n, d) in tqdm(enumerate(data), total=len(data)):
        messages = [
            {'role': 'system', 'content': config['summarize_prompt']},
            {'role': 'user', 'content': d['problem']},
            {'role': 'user', 'content': d['solution']}
        ]
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
        )

        summary = completion.choices[0].message.content
        print(f"Created summary:\n {summary}")

        messages = [
            {'role': 'system', 'content': config['infer_prompt']},
            {'role': 'user', 'content': d['problem'] + "\n" + summary},
        ]
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
        )
        cnt = completion.choices[0].message.content
        print(cnt)

        ans = find_boxed(cnt)
        ground_truth = d['answer']
        correctness = math_equal(ground_truth, ans)
        problem_cnt += 1
        if correctness:
            solved_cnt += 1
        print(f"The extracted answer is: {ans}")
        print(f"And the ground truth answer is: {ground_truth}")
        print(f"And the math comparison gives: {correctness}")

        if not config['test']:
            d['pred_summary'] = summary
            d['pred_steps'] = cnt
            d['pred_answer'] = ans
            d['correctness'] = correctness
            responses.append(d)

        if config['test'] and n >= 5:
            break

    # save the logs of the single run
    print("="*50)
    accuracy = float(solved_cnt) / float(problem_cnt)
    print(f"The test accuracy on MATH500 is {accuracy}")
    if not config['test']:
        save_jsonl(responses, prefix + "/responses.jsonl")
        logs['problems_count'] = problem_cnt
        logs["solved"] = solved_cnt
        logs['accuracy'] = accuracy
        with open(prefix + "/logs.json", "w", encoding='utf-8') as f:
            json.dump(logs, f, indent=4)
        print("Successfully run the full test, and relavent logs were saved.")
