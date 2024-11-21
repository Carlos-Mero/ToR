from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import TrainingArguments, Trainer
from utils import load_jsonl
from parser import strip_string
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime

def extract_training_data(config):
    data = []
    for path in config['training_dataset']:
        print(f"loading data form: {path}")
        for p in load_jsonl(path):
            p['answer'] = strip_string(p['answer'])
            messages = {
                "prompt": f"{config['infer_prompt']}\n\n{p['problem']}",
                "completion": f"### Idea\n\n{p['idea']}\n\n### Detailed Solution\n\n{p['solution']}"
            }
            data.append(messages)
    return Dataset.from_list(data)

def get_dataset(config):
    data = []
    for path in config['training_dataset']:
        print(f"loading data from: {path}")
        for p in load_jsonl(path):
            data.append(p)
    return data

def training_loop(config):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = config['model']
    # device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    training_data = extract_training_data(config)

    # response_template = "### Solution:"
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template=response_template,
    #     tokenizer=tokenizer
    # )
    training_args = SFTConfig(
        output_dir=config['log_dir'] + current_time,
        **config['sftparams']
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **config['lora_config']
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model,
        train_dataset = training_data,
        args = training_args,
        # formatting_func=formatting_prompts_func,
        # data_collator=collator,
        # peft_config=peft_config
    )

    trainer.train()
