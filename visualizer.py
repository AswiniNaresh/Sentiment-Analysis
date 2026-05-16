"""
visualizer.py — All plotting helpers for the sentiment analysis project.
"""
import os
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers / scripts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud

from src.config import LABEL_COLORS, OUTPUT_DIR, SENTIMENT_LABELS

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#e2e8f0",
    "grid.color":       "#334155",
    "font.family":      "monospace",
})

_COLORS = list(LABEL_COLORS.values())   # ["#ef4444", "#f59e0b", "#22c55e"]


# ─── 1. Label distribution ───────────────────────────────────────────────────

def plot_label_distribution(y, save: bool = True) -> str:
    import pandas as pd
    counts = pd.Series(y).map(SENTIMENT_LABELS).value_counts().reindex(
        list(SENTIMENT_LABELS.values()), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=_COLORS, width=0.5, zorder=3)
    ax.bar_label(bars, padding=4, color="#e2e8f0", fontsize=11)
    ax.set_title("Sentiment Label Distribution", fontsize=14, pad=12)
    ax.set_ylabel("Count")
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path


# ─── 2. Confusion matrix ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, save: bool = True) -> str:
    labels = list(SENTIMENT_LABELS.values())
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#334155",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path


# ─── 3. Cross-validation scores ──────────────────────────────────────────────

def plot_cv_scores(cv_scores: np.ndarray, save: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars  = ax.bar(folds, cv_scores, color="#38bdf8", width=0.5, zorder=3)
    ax.axhline(cv_scores.mean(), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Mean = {cv_scores.mean():.3f}")
    ax.bar_label(bars, fmt="%.3f", padding=4, color="#e2e8f0", fontsize=10)
    ax.set_title("Cross-Validation Accuracy per Fold", fontsize=14, pad=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cv_scores.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path


# ─── 4. Word cloud ───────────────────────────────────────────────────────────

def plot_wordcloud(texts: list[str], title: str = "Most Frequent Words",
                   save: bool = True) -> str:
    combined = " ".join(texts)
    wc = WordCloud(
        width=800, height=400,
        background_color="#0f172a",
        colormap="cool",
        max_words=150,
        collocations=False,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "wordcloud.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path


# ─── 5. VADER compound score distribution ────────────────────────────────────

def plot_vader_distribution(compound_scores, save: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(compound_scores, bins=30, color="#818cf8", edgecolor="#1e293b", zorder=3)
    ax.axvline(0.05,  color="#22c55e", linestyle="--", linewidth=1.5, label="Positive threshold")
    ax.axvline(-0.05, color="#ef4444", linestyle="--", linewidth=1.5, label="Negative threshold")
    ax.set_title("VADER Compound Score Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Compound Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "vader_distribution.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path
