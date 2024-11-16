from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import TrainingArguments, Trainer
from utils import load_jsonl
from parser import strip_string
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, TaskType
from datasets import Dataset
from tqdm import tqdm

def extract_training_data(config):
    data = []
    for path in config['training_dataset']:
        print(f"loading data form: {path}")
        for p in load_jsonl(path):
            p['answer'] = strip_string(p['answer'])
            messages = {
                "prompt": f"{config['infer_prompt']}\n{p['problem']}",
                "completion": f"{p['idea']}\n{p['solution']}"
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

def formatting_prompts_func(example):
    output_texts = []
    for p in tqdm(example):
        text = f"{p['infer_prompt']}\n{p['problem']}\n ### Solution\n{p['idea']}\n{p['solution']}"
        output_texts.append(text)
    return output_texts

def training_loop(config):
    model_name = config['model']
    # device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    training_data = extract_training_data(config)

    response_template = "### Solution:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    training_args = SFTConfig(
        output_dir=config['log_dir'],
        max_seq_length=2048,
        **config['sftparams']
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **config['lora_config']
    )

    trainer = SFTTrainer(
        model,
        train_dataset = training_data,
        args = training_args,
        # formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config
    )

    trainer.train()
