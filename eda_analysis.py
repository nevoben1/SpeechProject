import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

IEMOCAP_CSV = "iemocap_features_full.csv"
CREMAD_CSV  = "cremad_features_full.csv"

CONF_COLS     = ["conf_angry", "conf_disgusted", "conf_fearful",
                 "conf_happy", "conf_neutral", "conf_sad",
                 "conf_surprised", "conf_unknown"]
ACOUSTIC_COLS = ["mean_f0", "std_f0", "min_f0", "max_f0", "energy", "speech_rate"]


def load(path):
    df = pd.read_csv(path)
    df["e2v_correct"] = df["e2v_correct"].astype(str).str.strip().map(
        {"True": True, "False": False, "TRUE": True, "FALSE": False}
    )
    return df


# ─── Phase 1: Per-Class Error Profiling ──────────────────────────────────────

def plot_confusion_matrix(df, dataset_name):
    labels = sorted(df["ground_truth_4class"].unique())
    cm = confusion_matrix(df["ground_truth_4class"], df["e2v_4class"], labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(axes, [cm, cm_norm], ["Count", "Normalized"]):
        sns.heatmap(data, annot=True, fmt=".2f" if title == "Normalized" else "d",
                    xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
        ax.set_title(f"{dataset_name} — {title}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_{dataset_name.lower()}.png", dpi=150)
    plt.show()


def per_class_metrics(df, dataset_name):
    report = classification_report(df["ground_truth_4class"], df["e2v_4class"], output_dict=True)
    metrics = pd.DataFrame(report).T
    print(f"\n=== {dataset_name} Per-Class Metrics ===")
    print(metrics.round(3))
    return metrics


# ─── Phase 1: Confidence Distribution ────────────────────────────────────────

def plot_confidence_distribution(df, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for correct, label, color in [(True, "Correct", "steelblue"), (False, "Wrong", "tomato")]:
        subset = df[df["e2v_correct"] == correct]["e2v_conf_top"].dropna()
        axes[0].hist(subset, bins=30, alpha=0.6, label=label, color=color)
    axes[0].set_title(f"{dataset_name} — Confidence: Correct vs Wrong")
    axes[0].set_xlabel("e2v_conf_top")
    axes[0].legend()

    high_conf_wrong = df[(df["e2v_correct"] == False) & (df["e2v_conf_top"] >= 0.8)]
    counts = high_conf_wrong.groupby("ground_truth_4class").size()
    if not counts.empty:
        axes[1].bar(counts.index, counts.values, color="tomato")
    axes[1].set_title(f"{dataset_name} — High-Confidence Wrong (≥0.8) per True Class")
    axes[1].set_xlabel("True Label")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"confidence_{dataset_name.lower()}.png", dpi=150)
    plt.show()

    print(f"\n{dataset_name} — High-confidence wrong (≥0.8) top pairs:")
    print(
        high_conf_wrong.groupby(["ground_truth_4class", "e2v_4class"])
        .size().sort_values(ascending=False).head(10)
    )


def plot_softmax_heatmap(df, dataset_name):
    """Mean softmax scores per true class on misclassified samples — shows where probability mass leaks."""
    wrong = df[df["e2v_correct"] == False]
    available = [c for c in CONF_COLS if c in df.columns]
    if not available:
        return

    mean_conf = wrong.groupby("ground_truth_4class")[available].mean()
    plt.figure(figsize=(10, 5))
    sns.heatmap(mean_conf, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title(f"{dataset_name} — Mean Softmax Scores on Misclassified Samples")
    plt.xlabel("Class Confidence")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(f"softmax_heatmap_{dataset_name.lower()}.png", dpi=150)
    plt.show()


# ─── Phase 2: Speaker Analysis (IEMOCAP only) ────────────────────────────────

def speaker_accuracy(df):
    spk_acc = df.groupby("speaker_id")["e2v_correct"].mean().sort_values()
    spk_acc.plot(kind="bar", color="steelblue", figsize=(12, 4))
    plt.title("IEMOCAP — Per-Speaker Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("speaker_accuracy.png", dpi=150)
    plt.show()

    pivot = df.groupby(["speaker_id", "ground_truth_4class"])["e2v_correct"].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("IEMOCAP — Per-Speaker Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig("speaker_class_accuracy.png", dpi=150)
    plt.show()

    print("\nPer-speaker accuracy:")
    print(spk_acc.round(3))


# ─── Phase 3: Acoustic Feature Breakdown ─────────────────────────────────────

def acoustic_analysis(df, dataset_name):
    available = [c for c in ACOUSTIC_COLS if c in df.columns]
    if not available:
        print(f"{dataset_name}: no acoustic columns found")
        return

    df = df.copy()
    df["result"] = df["e2v_correct"].map({True: "Correct", False: "Wrong"})

    for col in available:
        plt.figure(figsize=(14, 5))
        df.boxplot(column=col, by=["ground_truth_4class", "result"])
        plt.title(f"{dataset_name} — {col} by Class and Result")
        plt.suptitle("")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"acoustic_{col}_{dataset_name.lower()}.png", dpi=150)
        plt.show()

    corr = df[available + ["e2v_correct"]].corr()["e2v_correct"].drop("e2v_correct")
    print(f"\n{dataset_name} — Acoustic feature correlation with correctness:")
    print(corr.sort_values().round(3))


# ─── Phase 4: Embedding Visualization — TODO ─────────────────────────────────

def embedding_visualization(df, dataset_name):
    pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df_iemocap = load(IEMOCAP_CSV)
    df_cremad  = load(CREMAD_CSV)

    print("IEMOCAP shape:", df_iemocap.shape)
    print("CREMA-D shape:", df_cremad.shape)

    # Phase 1
    for df, name in [(df_iemocap, "IEMOCAP"), (df_cremad, "CREMA-D")]:
        plot_confusion_matrix(df, name)
        per_class_metrics(df, name)
        plot_confidence_distribution(df, name)
        plot_softmax_heatmap(df, name)

    # Phase 2 — IEMOCAP only
    speaker_accuracy(df_iemocap)

    # Phase 3 — uncomment when ready
    # acoustic_analysis(df_iemocap, "IEMOCAP")
    # acoustic_analysis(df_cremad, "CREMA-D")


if __name__ == "__main__":
    main()
