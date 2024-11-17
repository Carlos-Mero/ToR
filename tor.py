import os
import json
import random
import argparse

from eval import run_cot, run_with_guidance, generate_ideas, run_cot_local, run_tor_local, run_lora_local
from train import training_loop

# Settings of the project
with open('./openai-config.json') as config_file:
    openai_config = json.load(config_file)
    os.environ["OPENAI_API_KEY"] = openai_config['openai_api_key']
    os.environ["OPENAI_BASE_URL"] = openai_config['openai_base_url']

def run(config):
    if config['type'] == 'generate':
        generate_ideas(config)
    elif config['type'] == 'guidance':
        run_with_guidance(config)
    elif config['type'] == 'basic':
        run_cot(config)
    elif config['type'] == 'basic-local':
        run_cot_local(config)
    elif config['type'] == 'tor-guidance-local':
        run_tor_local(config)
    elif config['type'] == 'train-lora':
        training_loop(config)
    elif config['type'] == 'lora-local':
        run_lora_local(config)
    else:
        raise NotImplementedError("Unknown inference strategy!")

def main():
    parser = argparse.ArgumentParser(description="This program is an attempt of Tree of Reasoning (ToR).")
    parser.add_argument('-c', '--config', type=str, help='path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1145, help='random seed for the program')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='The temperature parameter for inference')
    parser.add_argument('--test', action='store_true', help='Only complete the first five problems for test')
    # parser.add_argument('--generate', action='store_true', help='Using an advanced language model to generate the general reasoning steps.')
    # To be implemented, specific tor structure
    # parser.add_argument('--guidance', action='store_true', help="Enable summary guidance when doing inference")
    # parser.add_argument('--cot', action='store_true', help='Test the pass@1 accuracy with zero-shot cot')
    # parser.add_argument('--tor', action='store_true', help='Test the pass@1 accuracy with zero-shot tor reasoning')

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
        config['seed'] = args.seed
        config['temperature'] = args.temperature
        config['test'] = args.test
        random.seed(args.seed)
        run(config)
    print(f"program run with seed {args.seed}")

if __name__ == "__main__":
    main()
