import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

from transformers import pipeline, DataCollatorWithPadding

import DeepCore.deepcore.methods as methods

from typing import Literal, Union
import logging

logger = logging.getLogger("submitit.training")

SEED = 1337

Algo = Literal["bert-base-uncased", "roberta-base"]
Dataset = Literal["snli"]
Selection = Literal["uniform", "full"]

default_selection_args = dict(
    selection_epochs=40,
    selection_momentum=0.9,
    selection_weight_decay=5e-4,
    selection_optimizer="SGD",
    selection_nesterov=True,
    selection_lr=0.1,
    selection_test_interval=1,
    selection_test_fraction=1.0,
    balance=True,
)

unknown_args = dict(
    submodular="GraphCut",
    submodular_greedy="LazyGreedy",
    uncertainty="Entropy",
)


def train(
    algo: Algo = "bert-base-uncased",
    dataset_name: Dataset = "snli",
    epochs: int = 3,
    selection: str = "Uniform",
    fraction: float = 0.1,
    reduced: bool = False,
) -> int:
    logger.info(
        "Args received %s",
        {
            "algo": algo,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "selection": selection,
            "fraction": fraction,
            "reduced": reduced,
        },
    )

    dataset = load_dataset(dataset_name)
    model = AutoModelForSequenceClassification.from_pretrained(algo, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(algo)

    if reduced:
        dataset = dataset.shuffle(SEED)
        dataset["train"] = dataset["train"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(100))

    def encode(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

    dataset = dataset.filter(lambda batch: np.array(batch["label"]) != -1, batched=True)
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
        precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
        f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        seed=SEED,
        report_to="none",  # fix for some bug where tensorboard is added as an integration
    )

    # Coreset selection ========================================================

    # patch for selection method to access
    dataset["train"].classes = np.unique(dataset["train"]["label"])
    dataset["train"].targets = np.array(dataset["train"]["label"])

    selection_args = dict(
        epochs=default_selection_args["selection_epochs"],
        selection_method=unknown_args["uncertainty"],
        balance=default_selection_args["balance"],
        greedy=unknown_args["submodular_greedy"],
        function=unknown_args["submodular"],
    )
    method = methods.__dict__[selection](
        dataset["train"],
        default_selection_args,
        fraction,
        SEED,
        **selection_args,
    )
    subset = method.select()

    dataset["train"] = dataset["train"].select(subset["indices"])
    logger.info(f"Total size of training set: {len(dataset['train'])}")
    # =============================================================================

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.debug(f"Device in use: {args.device}")
    logger.info("==Training start==")
    x = trainer.train()
    logger.info("==Training completed==")
    logger.debug(x)

    ## Testing inference
    text = "This was a masterpiece"
    text2 = "Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    classifier([text, text2])
    with torch.no_grad():
        logits = model(**tokenizer(text, return_tensors="pt")).logits
        answer = logits.argmax().item()
        logger.debug(answer)

    ## Evaluation
    logger.info("==Evaluation start==")
    y = trainer.evaluate()
    logger.info("==Evaluation completed==")
    logger.debug(y)

    return answer


if __name__ == "__main__":
    train()
