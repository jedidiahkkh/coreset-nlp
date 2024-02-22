import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

from transformers import pipeline

SEED = 1337

def train():
    dataset = load_dataset("imdb")

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    args = TrainingArguments(
        output_dir="output", evaluation_strategy="epoch", num_train_epochs=1, seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"].shuffle(SEED).select(range(50)),
        eval_dataset=dataset["test"].shuffle(SEED).select(range(50)),
        compute_metrics=compute_metrics,
    )
    print(f"Device in use: {args.device}")
    x = trainer.train()
    print('==Training completed==')
    print(x)

    ## Testing inference
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    classifier(text)
    with torch.no_grad():
        logits = model(**tokenizer(text, return_tensors="pt")).logits
        print(logits.argmax().item())

    ## Evaluation
    print('==Begin evaluation==')
    y = trainer.evaluate()
    print('evaluation complete')
    print(y)


if __name__ == "__main__":
    train()
