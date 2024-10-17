import os
import json
import random
import argparse

from eval import run_cot

# Settings of the project
with open('./openai-config.json') as config_file:
    openai_config = json.load(config_file)
    os.environ["OPENAI_API_KEY"] = openai_config['openai_api_key']
    os.environ["OPENAI_BASE_URL"] = openai_config['openai_base_url']

def run(config):
    run_cot(config)

def main():
    parser = argparse.ArgumentParser(description="This program is an attempt of Tree of Reasoning (ToR).")
    parser.add_argument('-c', '--config', type=str, help='path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1145, help='random seed for the program')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='The temperature parameter for inference')
    # To be implemented, specific tor structure
    # parser.add_argument('--cot', action='store_true', help='Test the pass@1 accuracy with zero-shot cot')
    # parser.add_argument('--tor', action='store_true', help='Test the pass@1 accuracy with zero-shot tor reasoning')

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
        config['seed'] = args.seed
        config['temperature'] = args.temperature
        random.seed(args.seed)
        run(config)
    print(f"program run with seed {args.seed}")

if __name__ == "__main__":
    main()
