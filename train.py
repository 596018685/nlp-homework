import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from IPython.display import display, Markdown
import json, datasets
from tqdm import tqdm
from Datshelp import get_dataset
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
from peft import prepare_model_for_kbit_training
import bitsandbytes as bnb

device = torch.device('cuda:0')

def find_all_linear_nams(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def display_answer(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = inputs.to(device)
    pred = model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.1)
    res = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True).replace(text, "")
    display(Markdown(res))
    
def data_collator(features):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len) + ids[seq_len:] + [-100] * (longest - ids_l)
        )

        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels_list),
    }

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        self.model.save_pretrained(output_dir)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, #QLoRA 设计的 Double Quantization
    bnb_4bit_quant_type="nf4", #QLoRA 设计的 Normal Float 4 量化数据类型
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)
model_name_or_path = "/root/model/Baichuan2-13B-Chat"
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                             device_map="auto", 
                                             quantization_config=bnb_config, trust_remote_code=True)

train_set=get_dataset(data_load="/root/data/yuanshen.json").load_from_disk("yuanshen_data")
model = prepare_model_for_kbit_training(model)

lora_modules = find_all_linear_nams(model)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=lora_modules,
)

model = get_peft_model(model, peft_config)
model.supports_gradient_checkpointing = True
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model.isparallelizable = True
model.model_parallel = True
tokenizer.pad_token_id = config.pad_token_id 

batch_size = 6
train_args = TrainingArguments(learning_rate=1e-4, 
                               per_device_train_batch_size=batch_size, 
                               gradient_accumulation_steps=10,
                               max_steps=600,
                               save_steps=100,
                               logging_steps=10,
                               output_dir="baichuan2-13b-lora",
                               remove_unused_columns=False,
                              )

trainer = ModifiedTrainer(
    model=model,
    train_dataset=train_set,
    args=train_args,
    data_collator=data_collator,
)
trainer.train()
model.save_pretrained("/root/model/checkpoint/output")


