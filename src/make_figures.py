"""
Build all report figures from the JSON files produced by run_experiments.py.

Everything here is generated from logged results, so the figures always match
the numbers in results/experiments.json. Output goes to results/figures/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_json(name):
    path = RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"saved {name}")


def class_distribution(stats):
    counts = stats["class_counts"]["train"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.keys(), counts.values(), color="#4878a8")
    ax.set_title("Class distribution (training set)")
    ax.set_ylabel("examples")
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    save(fig, "class_distribution.png")


def lr_comparison(store):
    runs = {rid: r for rid, r in store.items()
            if rid.startswith("distilbert_lr") and rid.endswith("_b32_e3")}
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for rid, r in sorted(runs.items(), key=lambda x: x[1]["lr"]):
        epochs = [h["epoch"] for h in r["history"]]
        f1 = [h["eval_macro_f1"] for h in r["history"]]
        ax.plot(epochs, f1, marker="o", label=f"lr={r['lr']:.0e}")
    ax.set_title("Learning rate search (DistilBERT, batch 32)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "lr_comparison.png")


def batch_comparison(store):
    pairs = {}
    for rid, r in store.items():
        if rid.startswith("distilbert_lr") and rid.endswith("_e3") and not r["use_weights"]:
            pairs[r["batch"]] = max(pairs.get(r["batch"], 0), r["best_val_macro_f1"])
    if len(pairs) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = sorted(pairs)
    ax.bar([str(k) for k in keys], [pairs[k] for k in keys],
           color=["#4878a8", "#e49444"], width=0.5)
    for i, k in enumerate(keys):
        ax.text(i, pairs[k], f"{pairs[k]:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Batch size (best lr, DistilBERT)")
    ax.set_xlabel("batch size")
    ax.set_ylabel("best validation macro-F1")
    ax.set_ylim(0, 1)
    save(fig, "batch_comparison.png")


def epochs_curve(store):
    run = next((r for rid, r in store.items() if rid.endswith("_e5")), None)
    if run is None:
        return
    epochs = [h["epoch"] for h in run["history"]]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [h["train_loss"] for h in run["history"]],
             marker="o", color="#4878a8", label="train loss")
    ax1.plot(epochs, [h["eval_loss"] for h in run["history"]],
             marker="s", color="#d1615d", label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_xticks(epochs)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["eval_macro_f1"] for h in run["history"]],
             marker="^", color="#6a9f58", label="validation macro-F1")
    ax2.set_ylabel("macro-F1")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend below the axes so it never sits on top of the curves
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, -0.15), ncols=3, frameon=False)
    ax1.set_title("Training for 5 epochs (best DistilBERT config)")
    ax1.grid(alpha=0.3)
    save(fig, "epochs_curve.png")


def class_weights_effect(store):
    weighted = next((r for rid, r in store.items()
                     if rid.endswith("_weighted")), None)
    if weighted is None:
        return
    base_id = weighted["run_id"].replace("_weighted", "")
    base = store.get(base_id)
    if base is None:
        return
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [base["val_per_class_f1"][l] for l in LABELS],
           width=0.4, label="plain loss", color="#4878a8")
    ax.bar(x + 0.2, [weighted["val_per_class_f1"][l] for l in LABELS],
           width=0.4, label="class-weighted loss", color="#e49444")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("validation F1")
    ax.set_title("Effect of class-weighted loss per class (DistilBERT)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save(fig, "class_weights_effect.png")


def model_comparison(store):
    finals = {m: store.get(f"final_{m}") for m in ("distilbert", "bert")}
    if not all(finals.values()):
        return
    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    names = ["accuracy", "macro-F1", "weighted-F1"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, color) in enumerate([("distilbert", "#4878a8"), ("bert", "#e49444")]):
        vals = [finals[m][k] for k in metrics]
        ax.bar(x + (i - 0.5) * 0.35, vals, width=0.35, label=m, color=color)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 0.5) * 0.35, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)  # headroom so the legend clears the bars
    ax.set_title("Final test-set comparison")
    ax.legend(loc="upper right", ncols=2)
    save(fig, "model_comparison.png")


def confusion_matrices(store):
    for m in ("distilbert", "bert"):
        r = store.get(f"final_{m}")
        if r is None or "confusion_matrix" not in r:
            continue
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6), LABELS, rotation=45, ha="right")
        ax.set_yticks(range(6), LABELS)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix — {m} (test set)")
        fig.colorbar(im, shrink=0.8)
        save(fig, f"confusion_matrix_{m}.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    store = load_json("experiments.json") or {}
    stats = load_json("dataset_stats.json")
    if stats:
        class_distribution(stats)
    lr_comparison(store)
    batch_comparison(store)
    epochs_curve(store)
    class_weights_effect(store)
    model_comparison(store)
    confusion_matrices(store)
    print("figures done")


if __name__ == "__main__":
    main()
