import json, datasets
from tqdm import tqdm

def preprocess(tokenizer, config, file_path, max_seq_length, prompt_key, target_key, skip_overlength=False):
    with open(file_path, "r", encoding="utf8") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            prompt_ids = tokenizer.encode(example[prompt_key], max_length=max_seq_length, truncation=True)
            target_ids = tokenizer.encode(example[target_key], max_length=max_seq_length, truncation=True)
            input_ids = prompt_ids + target_ids + [config.eos_token_id]
            if skip_overlength and len(input_ids) > max_seq_length:
                continue
            input_ids = input_ids[:max_seq_length]
            yield {
                "input_ids": input_ids,
                "seq_len": len(prompt_ids)
            }

def get_dataset(data_load="/root/data/yuanshen.json"):
    dataset = datasets.Dataset.from_generator(lambda: preprocess(tokenizer, 
                                                config, 
                                                data_load, 
                                                max_seq_length=2000, 
                                                prompt_key="q",
                                                target_key="a",))

    dataset.save_to_disk("yuanshen_data")
    return dataset
