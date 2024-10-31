from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from utils import load_jsonl, save_jsonl
from parser import find_box, strip_string

def extract_trainingset_tokens(config, tokenizer):
    data = []
    device = "cuda" # the device to load the model onto
    for path in config['datasets']:
        print(f"loading data form: {path}")
        for p in load_jsonl(path):
            p['answer'] = strip_string(p['answer'])
            messages = [
                {"role": "system", "content": config['infer_prompt']},
                {"role": "user", "content": p['problem']}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            p['tokenized_inputs'] = model_inputs
            data.append(p)
    return data

def training_loop(config):
    model_name = config['model']
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    training_data = extract_trainingset_tokens(config, tokenizer)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
