import ast
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu

IEMOCAP_CSV = "iemocap_features_full.csv"
CREMAD_CSV  = "cremad_features_full.csv"

ACOUSTIC_COLS = ["mean_f0", "std_f0", "energy", "speech_rate"]


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



# ─── Phase 2: Speaker Analysis (IEMOCAP only) ────────────────────────────────

def speaker_accuracy(df):
    pivot = df.groupby(["speaker_id", "ground_truth_4class"])["e2v_correct"].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("IEMOCAP — Per-Speaker Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig("speaker_class_accuracy.png", dpi=150)
    plt.show()


# ─── Phase 3: Acoustic Feature Breakdown ─────────────────────────────────────

def acoustic_analysis(df, dataset_name):
    available = [c for c in ACOUSTIC_COLS if c in df.columns]
    if not available:
        print(f"{dataset_name}: no acoustic columns found")
        return

    df = df.copy()
    df["result"] = df["e2v_correct"].map({True: "Correct", False: "Wrong"})
    classes = sorted(df["ground_truth_4class"].unique())

    for col in available:
        fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 5), sharey=True)
        if len(classes) == 1:
            axes = [axes]
        for ax, cls in zip(axes, classes):
            subset = df[df["ground_truth_4class"] == cls]
            sns.violinplot(data=subset, x="result", y=col, order=["Correct", "Wrong"],
                           palette={"Correct": "steelblue", "Wrong": "tomato"},
                           inner="box", ax=ax)
            ax.set_title(cls)
            ax.set_xlabel("")
            if ax != axes[0]:
                ax.set_ylabel("")
        fig.suptitle(f"{dataset_name} — {col} by Class and Result", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"acoustic_{col}_{dataset_name.lower()}.png", dpi=150)
        plt.show()

    corr = df[available + ["e2v_correct"]].corr()["e2v_correct"].drop("e2v_correct")
    print(f"\n{dataset_name} — Acoustic feature correlation with correctness:")
    print(corr.sort_values().round(3))

# ─── Phase 4: Embedding Visualization ───────────────────────────────────────

def embedding_visualization(df, dataset_name):
    if "embedding" not in df.columns:
        print(f"{dataset_name}: no embedding column found")
        return

    df = df.copy()
    embeddings = np.array(df["embedding"].apply(ast.literal_eval).tolist())

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_

    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    df["result"] = df["e2v_correct"].map({True: "Correct", False: "Wrong"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1 — colored by true class
    classes = sorted(df["ground_truth_4class"].unique())
    palette = sns.color_palette("tab10", len(classes))
    for cls, color in zip(classes, palette):
        mask = df["ground_truth_4class"] == cls
        axes[0].scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                        label=cls, color=color, alpha=0.5, s=15)
    axes[0].set_title(f"{dataset_name} — Embeddings by True Class")
    axes[0].set_xlabel(f"PC1 ({var[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({var[1]:.1%} var)")
    axes[0].legend(markerscale=2)

    # Plot 2 — colored by correct/wrong
    colors = {"Correct": "steelblue", "Wrong": "tomato"}
    for result, color in colors.items():
        mask = df["result"] == result
        axes[1].scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                        label=result, color=color, alpha=0.4, s=15)
    axes[1].set_title(f"{dataset_name} — Embeddings by Prediction Outcome")
    axes[1].set_xlabel(f"PC1 ({var[0]:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({var[1]:.1%} var)")
    axes[1].legend(markerscale=2)

    plt.tight_layout()
    plt.savefig(f"embedding_pca_{dataset_name.lower()}.png", dpi=150)
    plt.show()
    print(f"{dataset_name} — PCA variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}")



# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df_iemocap = load(IEMOCAP_CSV)
    df_cremad  = load(CREMAD_CSV)

    print("IEMOCAP shape:", df_iemocap.shape)
    print("CREMA-D shape:", df_cremad.shape)

    # Phase 1
    for df, name in [(df_iemocap, "IEMOCAP"), (df_cremad, "CREMA-D")]:
        plot_confusion_matrix(df, name)
        plot_confidence_distribution(df, name)

    # Phase 2 — IEMOCAP only
    speaker_accuracy(df_iemocap)

    # Phase 3
    acoustic_analysis(df_iemocap, "IEMOCAP")
    acoustic_analysis(df_cremad, "CREMA-D")

    # Phase 4
    embedding_visualization(df_iemocap, "IEMOCAP")
    embedding_visualization(df_cremad, "CREMA-D")



if __name__ == "__main__":
    main()
