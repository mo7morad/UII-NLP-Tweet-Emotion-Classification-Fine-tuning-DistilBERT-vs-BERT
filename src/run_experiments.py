"""
Fine-tuning experiments for emotion classification (dair-ai/emotion).

The script compares two pre-trained language models (DistilBERT and BERT-base)
under different hyperparameters: learning rate, batch size and number of epochs.
Every run is logged to results/experiments.json as soon as it finishes, so the
script can be stopped and restarted without repeating completed runs.

Experiment plan
---------------
Stage A  - learning rate search on DistilBERT (2e-5, 3e-5, 5e-5), batch 32, 3 epochs
Stage B  - batch size check (16 vs 32) at the best learning rate
Stage C  - longer run (5 epochs) with the best config to study over/underfitting
Stage D  - class-weighted loss vs plain loss (the dataset is imbalanced)
Stage E  - BERT-base with the best config + one alternative learning rate
Stage F  - final runs: best setup per model, 4 epochs, keep the best epoch by
           validation macro-F1, then evaluate ONCE on the held-out test set

The validation set is used for every tuning decision. The test set is only
touched by the two final runs.

Usage
-----
    python src/run_experiments.py            # full suite (~30-45 min on a GPU)
    QUICK=1 python src/run_experiments.py    # small sanity run (a few minutes)
    SMOKE=1 python src/run_experiments.py    # offline pipeline test, tiny random
                                             # model + synthetic data, no downloads
"""

import gc
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------- constants --

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CKPT_DIR = ROOT / "checkpoints"          # temporary, deleted after each run
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.json"

SEED = 42
MAX_LEN = 64                              # covers >99% of tweets (checked below)
TRAIN_SIZE = 8000                         # stratified subsample of the 16k train split
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

QUICK = os.environ.get("QUICK") == "1"
SMOKE = os.environ.get("SMOKE") == "1"

MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


@dataclass
class RunConfig:
    run_id: str
    model: str            # key into MODELS (or "smoke")
    lr: float
    batch: int
    epochs: int
    use_weights: bool = False
    final: bool = False   # final runs also evaluate on the test set


# ------------------------------------------------------------------- data ----

def load_data():
    """Load the emotion dataset, subsample the train split and save CSV copies."""
    if SMOKE:
        return _synthetic_data()

    ds = load_dataset("dair-ai/emotion")

    train_n = 500 if QUICK else TRAIN_SIZE
    ds["train"] = ds["train"].train_test_split(
        train_size=train_n, stratify_by_column="label", seed=SEED
    )["train"]
    if QUICK:
        ds["validation"] = ds["validation"].select(range(300))
        ds["test"] = ds["test"].select(range(300))

    DATA_DIR.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        ds[split].to_pandas().to_csv(DATA_DIR / f"{split}.csv", index=False)

    return ds


def _synthetic_data():
    """Tiny fake dataset with the same schema, for offline pipeline testing."""
    rng = np.random.default_rng(SEED)
    words = {
        0: "sad down crying lost", 1: "happy great joy fun",
        2: "love dear heart sweet", 3: "angry mad furious rage",
        4: "scared afraid worry fear", 5: "wow sudden shock surprised",
    }
    texts, labels = [], []
    for _ in range(600):
        y = int(rng.integers(0, 6))
        pool = words[y].split()
        texts.append("i feel " + " ".join(rng.choice(pool, size=4)))
        labels.append(y)
    full = Dataset.from_dict({"text": texts, "label": labels})
    full = full.cast_column("label", ClassLabel(names=LABELS))
    split1 = full.train_test_split(test_size=200, seed=SEED, stratify_by_column="label")
    split2 = split1["test"].train_test_split(test_size=100, seed=SEED, stratify_by_column="label")
    return DatasetDict(train=split1["train"], validation=split2["train"], test=split2["test"])


def dataset_stats(ds, tokenizer):
    """Record sizes, class balance and token-length percentiles (justifies MAX_LEN)."""
    lengths = [len(x) for x in tokenizer(list(ds["train"]["text"]), truncation=False)["input_ids"]]
    counts = {}
    for split in ("train", "validation", "test"):
        labels = list(ds[split]["label"])
        counts[split] = {LABELS[i]: int(np.sum(np.array(labels) == i)) for i in range(6)}
    stats = {
        "sizes": {s: len(ds[s]) for s in ("train", "validation", "test")},
        "class_counts": counts,
        "token_length": {
            "mean": float(np.mean(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(np.max(lengths)),
        },
        "max_len_used": MAX_LEN,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ------------------------------------------------------------- model setup ---

def get_tokenizer(model_key):
    if model_key == "smoke":
        return _smoke_tokenizer()
    return AutoTokenizer.from_pretrained(MODELS[model_key])


def new_model(model_key):
    if model_key == "smoke":
        return _smoke_model()
    return AutoModelForSequenceClassification.from_pretrained(
        MODELS[model_key],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)},
    )


def _smoke_tokenizer():
    from transformers import BertTokenizerFast
    vocab_dir = ROOT / "checkpoints" / "smoke_vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "feel"]
    for line in ["sad down crying lost", "happy great joy fun", "love dear heart sweet",
                 "angry mad furious rage", "scared afraid worry fear", "wow sudden shock surprised"]:
        vocab += line.split()
    (vocab_dir / "vocab.txt").write_text("\n".join(vocab))
    return BertTokenizerFast(vocab_file=str(vocab_dir / "vocab.txt"), do_lower_case=True)


def _smoke_model():
    from transformers import BertConfig, BertForSequenceClassification
    config = BertConfig(
        vocab_size=40, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, num_labels=len(LABELS),
    )
    return BertForSequenceClassification(config)


def tokenize_dataset(ds, tokenizer):
    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)
    return ds.map(encode, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------- training ---

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


class WeightedLossTrainer(Trainer):
    """Trainer with an optional class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def load_store():
    if EXPERIMENTS_FILE.exists():
        return json.loads(EXPERIMENTS_FILE.read_text())
    return {}


def save_store(store):
    RESULTS_DIR.mkdir(exist_ok=True)
    tmp = EXPERIMENTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.replace(EXPERIMENTS_FILE)


def run_experiment(cfg: RunConfig, ds_tok, tokenizer, raw_test_texts, class_weights, store):
    """Fine-tune one configuration and record everything we need for the report."""
    if cfg.run_id in store:
        print(f"[skip] {cfg.run_id} already done")
        return store[cfg.run_id]

    print(f"\n===== {cfg.run_id} | lr={cfg.lr} batch={cfg.batch} "
          f"epochs={cfg.epochs} weights={cfg.use_weights} =====")
    set_seed(SEED)
    model = new_model(cfg.model)

    args = TrainingArguments(
        output_dir=str(CKPT_DIR / cfg.run_id),
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch,
        per_device_eval_batch_size=64,
        num_train_epochs=cfg.epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch" if cfg.final else "no",
        load_best_model_at_end=cfg.final,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        weight_decay=0.01,
        seed=SEED,
        fp16=(DEVICE == "cuda"),
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights if cfg.use_weights else None,
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    runtime = time.time() - start

    # per-epoch history from the trainer logs
    history, train_loss_by_epoch = [], {}
    for log in trainer.state.log_history:
        if "loss" in log and "eval_loss" not in log:
            train_loss_by_epoch[round(log["epoch"])] = log["loss"]
        if "eval_macro_f1" in log:
            history.append({
                "epoch": round(log["epoch"]),
                "train_loss": train_loss_by_epoch.get(round(log["epoch"])),
                "eval_loss": log["eval_loss"],
                "eval_accuracy": log["eval_accuracy"],
                "eval_macro_f1": log["eval_macro_f1"],
            })
    best = max(history, key=lambda h: h["eval_macro_f1"])

    # per-class F1 on validation (for the imbalance analysis)
    val_pred = trainer.predict(ds_tok["validation"])
    val_report = classification_report(
        val_pred.label_ids, val_pred.predictions.argmax(-1),
        target_names=LABELS, output_dict=True, zero_division=0,
    )

    record = {
        **asdict(cfg),
        "device": DEVICE,
        "runtime_sec": round(runtime, 1),
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["eval_macro_f1"],
        "val_per_class_f1": {l: val_report[l]["f1-score"] for l in LABELS},
    }

    if cfg.final:
        test_pred = trainer.predict(ds_tok["test"])
        y_true, y_pred = test_pred.label_ids, test_pred.predictions.argmax(-1)
        record["test_accuracy"] = accuracy_score(y_true, y_pred)
        record["test_macro_f1"] = f1_score(y_true, y_pred, average="macro")
        record["test_weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
        record["test_report"] = classification_report(
            y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
        record["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        errors = [
            {"text": raw_test_texts[i], "true": LABELS[y_true[i]], "pred": LABELS[y_pred[i]]}
            for i in range(len(y_true)) if y_true[i] != y_pred[i]
        ]
        pd.DataFrame(errors[:25]).to_csv(
            RESULTS_DIR / f"errors_{cfg.model}.csv", index=False)

    store[cfg.run_id] = record
    save_store(store)

    # free memory and clean up checkpoints before the next run
    del trainer, model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
    shutil.rmtree(CKPT_DIR / cfg.run_id, ignore_errors=True)

    print(f"[done] {cfg.run_id}: best val macro-F1 {record['best_val_macro_f1']:.4f} "
          f"(epoch {record['best_epoch']}) in {runtime/60:.1f} min")
    return record


# -------------------------------------------------------------------- main ---

def main():
    print(f"Device: {DEVICE} | quick={QUICK} smoke={SMOKE}")
    t0 = time.time()

    model_keys = ["smoke"] if SMOKE else ["distilbert", "bert"]
    primary = model_keys[0]

    ds = load_data()
    tokenizers = {k: get_tokenizer(k) for k in model_keys}
    stats = dataset_stats(ds, tokenizers[primary])
    print(f"Dataset: {stats['sizes']} | token length p99 = {stats['token_length']['p99']}")

    # tokenized copies per model (their tokenizers differ)
    tokenized = {key: tokenize_dataset(ds, tokenizers[key]) for key in model_keys}

    train_labels = np.array(list(ds["train"]["label"]))
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_labels),
        dtype=torch.float32,
    )
    raw_test_texts = list(ds["test"]["text"])
    store = load_store()

    def run(cfg):
        return run_experiment(cfg, tokenized[cfg.model], tokenizers[cfg.model],
                              raw_test_texts, class_weights, store)

    if SMOKE or QUICK:
        # reduced plan: one tuning run + one final run, just to prove the pipeline
        run(RunConfig("tune_smoke", primary, 2e-5 if not SMOKE else 1e-3, 32, 1))
        run(RunConfig("final_smoke", primary, 2e-5 if not SMOKE else 1e-3, 32, 1, final=True))
        print(f"\nPipeline OK in {(time.time()-t0)/60:.1f} min. "
              f"Results in {EXPERIMENTS_FILE}")
        (RESULTS_DIR / "ALL_DONE.txt").write_text("smoke/quick run finished\n")
        return

    # Stage A: learning rate search (DistilBERT)
    for lr in (2e-5, 3e-5, 5e-5):
        run(RunConfig(f"distilbert_lr{lr}_b32_e3", "distilbert", lr, 32, 3))
    best_lr = max((2e-5, 3e-5, 5e-5),
                  key=lambda lr: store[f"distilbert_lr{lr}_b32_e3"]["best_val_macro_f1"])
    print(f"\n>> best learning rate: {best_lr}")

    # Stage B: batch size at the best learning rate
    run(RunConfig(f"distilbert_lr{best_lr}_b16_e3", "distilbert", best_lr, 16, 3))
    f1_b16 = store[f"distilbert_lr{best_lr}_b16_e3"]["best_val_macro_f1"]
    f1_b32 = store[f"distilbert_lr{best_lr}_b32_e3"]["best_val_macro_f1"]
    best_batch = 16 if f1_b16 > f1_b32 else 32
    print(f">> best batch size: {best_batch}")

    # Stage C: 5-epoch run to look at overfitting
    run(RunConfig(f"distilbert_lr{best_lr}_b{best_batch}_e5", "distilbert",
                  best_lr, best_batch, 5))

    # Stage D: class-weighted loss
    run(RunConfig(f"distilbert_lr{best_lr}_b{best_batch}_e3_weighted", "distilbert",
                  best_lr, best_batch, 3, use_weights=True))
    baseline_f1 = store[f"distilbert_lr{best_lr}_b{best_batch}_e3"]["best_val_macro_f1"]
    weighted_f1 = store[f"distilbert_lr{best_lr}_b{best_batch}_e3_weighted"]["best_val_macro_f1"]
    use_weights_final = weighted_f1 > baseline_f1
    print(f">> class weights help: {use_weights_final} "
          f"(weighted {weighted_f1:.4f} vs plain {baseline_f1:.4f})")

    # Stage E: BERT-base at the best config + one alternative learning rate
    alt_lr = 2e-5 if best_lr != 2e-5 else 3e-5
    run(RunConfig(f"bert_lr{best_lr}_b{best_batch}_e3", "bert", best_lr, best_batch, 3))
    run(RunConfig(f"bert_lr{alt_lr}_b{best_batch}_e3", "bert", alt_lr, best_batch, 3))
    bert_lr = max((best_lr, alt_lr),
                  key=lambda lr: store[f"bert_lr{lr}_b{best_batch}_e3"]["best_val_macro_f1"])
    print(f">> best learning rate for bert: {bert_lr}")

    # Stage F: final runs, test set evaluated exactly once per model
    run(RunConfig("final_distilbert", "distilbert", best_lr, best_batch, 4,
                  use_weights=use_weights_final, final=True))
    run(RunConfig("final_bert", "bert", bert_lr, best_batch, 4,
                  use_weights=use_weights_final, final=True))

    summary = {
        m: {
            "config": {k: store[f"final_{m}"][k]
                       for k in ("lr", "batch", "epochs", "use_weights", "best_epoch")},
            "test_accuracy": store[f"final_{m}"]["test_accuracy"],
            "test_macro_f1": store[f"final_{m}"]["test_macro_f1"],
            "test_weighted_f1": store[f"final_{m}"]["test_weighted_f1"],
        }
        for m in ("distilbert", "bert")
    }
    (RESULTS_DIR / "test_results.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "ALL_DONE.txt").write_text(
        time.strftime("finished %Y-%m-%d %H:%M:%S\n"))

    print(f"\nAll experiments finished in {(time.time()-t0)/60:.1f} min.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
