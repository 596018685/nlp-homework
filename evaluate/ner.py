import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import DatasetDict
from transformers import pipeline

def process_function(examples):
    tokenized_exmaples = tokenizer(examples["tokens"], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_exmaples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_exmaples["labels"] = labels
    return tokenized_exmaples

def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels) 
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels) 
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")
    return {
        "f1": result["overall_f1"]
    }

ner_datasets = DatasetDict.load_from_disk("ner_data")
label_list = ner_datasets["train"].features["ner_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained("/root/model/chinese-macbert-base")
tokenizer(ner_datasets["train"][0]["tokens"], is_split_into_words=True)
tokenized_datasets = ner_datasets.map(process_function, batched=True)
model = AutoModelForTokenClassification.from_pretrained("/root/model/chinese-macbert-base", num_labels=len(label_list))
seqeval = evaluate.load("seqeval_metric.py")

args = TrainingArguments(
    output_dir="models_for_ner",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_steps=50,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")
result1=input("你想评估的生成")
result=ner_pipe(result1)
ner_result = {}

for r in result:
    if r["entity_group"] not in ner_result:
        ner_result[r["entity_group"]] = []
    ner_result[r["entity_group"]].append(x[r["start"]: r["end"]])

print(ner_result)