from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import DEFAULT_MODEL_NAME, ID2LABEL, LABEL2ID, ensure_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on AG News.")
    parser.add_argument("--dataset_name", type=str, default="sh0416/ag_news")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=str(Path("models") / "news-topic-bert"))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--do_train", action="store_true", help="Train the model.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_model_dir(args.output_dir)

    dataset = load_dataset(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_columns = dataset["train"].column_names

    if "label" in train_columns:
        label_column = "label"
    elif "Class Index" in train_columns:
        label_column = "Class Index"
    else:
        raise ValueError(
            f"Could not find a label column in dataset columns: {train_columns}"
        )

    if "text" in train_columns:
        def build_text(batch: dict[str, list[str]]) -> list[str]:
            return batch["text"]
    elif "title" in train_columns and "description" in train_columns:
        def build_text(batch: dict[str, list[str]]) -> list[str]:
            return [
                f"{title}. {description}".strip()
                for title, description in zip(batch["title"], batch["description"])
            ]
    else:
        fallback_text_column = next(
            (name for name in train_columns if name != label_column),
            None,
        )
        if fallback_text_column is None:
            raise ValueError(
                f"Could not infer a text column from dataset columns: {train_columns}"
            )

        def build_text(batch: dict[str, list[str]]) -> list[str]:
            return batch[fallback_text_column]

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(build_text(batch), truncation=True, max_length=args.max_length)

    raw_train_labels = dataset["train"][label_column]
    if len(raw_train_labels) == 0:
        raise ValueError("Training split is empty; cannot infer label mapping.")

    min_label = int(min(raw_train_labels))
    max_label = int(max(raw_train_labels))
    label_offset = 1 if min_label == 1 and max_label == len(ID2LABEL) else 0

    remove_columns = [column for column in train_columns if column != label_column]
    tokenized = dataset.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized = tokenized.rename_column(label_column, "labels")
    if label_offset != 0:
        tokenized = tokenized.map(lambda batch: {"labels": [label - label_offset for label in batch["labels"]]}, batched=True)
    tokenized.set_format("torch")

    train_dataset = tokenized["train"]
    eval_split_name = "test" if "test" in tokenized else "validation"
    eval_dataset = tokenized[eval_split_name]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.do_eval else None,
    )

    if args.do_train:
        trainer.train()

    metrics = {}
    if args.do_eval:
        metrics = trainer.evaluate()
        print(json.dumps(metrics, indent=2))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary_path = output_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
