"""Multinomial Naive Bayes text classification on synthetic course data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z_]+")


CLASS_KEYWORDS = {
    "algorithm": ["matrix", "gradient", "loss", "model", "feature", "training", "weight", "vector"],
    "robotics": ["robot", "sensor", "motion", "joint", "path", "control", "arm", "navigation"],
    "vision": ["image", "pixel", "camera", "filter", "edge", "object", "segmentation", "texture"],
}

COMMON_WORDS = ["system", "data", "method", "result", "sample", "analysis", "project", "task"]


@dataclass
class Dataset:
    train_texts: list[str]
    train_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]


@dataclass
class NBModel:
    classes: list[str]
    vocabulary: list[str]
    log_prior: dict[str, float]
    log_likelihood: dict[str, np.ndarray]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def make_sentence(label: str, rng: np.random.Generator) -> str:
    own = CLASS_KEYWORDS[label]
    other_labels = [name for name in CLASS_KEYWORDS if name != label]
    words = []
    words.extend(rng.choice(own, size=5, replace=True).tolist())
    words.extend(rng.choice(COMMON_WORDS, size=3, replace=True).tolist())
    if rng.random() < 0.38:
        words.append(str(rng.choice(CLASS_KEYWORDS[str(rng.choice(other_labels))])))
    rng.shuffle(words)
    return " ".join(words)


def make_dataset(seed: int = 917, train_per_class: int = 80, test_per_class: int = 35) -> Dataset:
    rng = np.random.default_rng(seed)
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for label in CLASS_KEYWORDS:
        for _ in range(train_per_class):
            train_texts.append(make_sentence(label, rng))
            train_labels.append(label)
        for _ in range(test_per_class):
            test_texts.append(make_sentence(label, rng))
            test_labels.append(label)
    return Dataset(train_texts, train_labels, test_texts, test_labels)


def build_vocabulary(texts: list[str]) -> list[str]:
    counts = Counter(token for text in texts for token in tokenize(text))
    return sorted(counts)


def vectorize(texts: list[str], vocabulary: list[str]) -> np.ndarray:
    index = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=float)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            if token in index:
                matrix[row, index[token]] += 1.0
    return matrix


def fit_naive_bayes(texts: list[str], labels: list[str], alpha: float = 1.0) -> NBModel:
    classes = sorted(set(labels))
    vocabulary = build_vocabulary(texts)
    x = vectorize(texts, vocabulary)
    label_array = np.array(labels)
    log_prior: dict[str, float] = {}
    log_likelihood: dict[str, np.ndarray] = {}
    for cls in classes:
        mask = label_array == cls
        class_count = float(np.sum(mask))
        word_counts = x[mask].sum(axis=0) + alpha
        log_prior[cls] = math.log(class_count / len(labels))
        log_likelihood[cls] = np.log(word_counts / word_counts.sum())
    return NBModel(classes, vocabulary, log_prior, log_likelihood)


def predict(model: NBModel, texts: list[str]) -> list[str]:
    x = vectorize(texts, model.vocabulary)
    predictions = []
    for row in x:
        scores = {
            cls: model.log_prior[cls] + float(row @ model.log_likelihood[cls])
            for cls in model.classes
        }
        predictions.append(max(scores, key=scores.get))
    return predictions


def confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str]) -> np.ndarray:
    index = {cls: i for i, cls in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[index[truth], index[pred]] += 1
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    scores = []
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def plot_confusion(matrix: np.ndarray, classes: list[str], output: Path) -> Path:
    path = output / "nb_confusion_matrix.png"
    plt.figure(figsize=(6.2, 5.4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="samples")
    plt.xticks(range(len(classes)), classes, rotation=25, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("Naive Bayes confusion matrix")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x, y, str(matrix[y, x]), ha="center", va="center", color="#111827")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_top_words(model: NBModel, output: Path, topn: int = 6) -> Path:
    path = output / "nb_top_words.png"
    fig, axes = plt.subplots(1, len(model.classes), figsize=(11.5, 4.2), sharey=True)
    vocab = np.array(model.vocabulary)
    for ax, cls in zip(axes, model.classes):
        values = model.log_likelihood[cls]
        top = np.argsort(values)[-topn:][::-1]
        ax.barh(vocab[top][::-1], values[top][::-1], color="#2f80ed")
        ax.set_title(cls)
        ax.grid(axis="x", linestyle="--", alpha=0.28)
    fig.suptitle("Top likelihood words by class")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_metrics(accuracy: float, f1: float, output: Path) -> Path:
    path = output / "nb_metric_summary.png"
    plt.figure(figsize=(6.6, 4.5))
    plt.bar(["accuracy", "macro F1"], [accuracy, f1], color=["#27ae60", "#2f80ed"])
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Text classification evaluation")
    plt.grid(axis="y", linestyle="--", alpha=0.28)
    for i, score in enumerate([accuracy, f1]):
        plt.text(i, score + 0.025, f"{score:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def run(output: Path, seed: int = 917) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    data = make_dataset(seed=seed)
    model = fit_naive_bayes(data.train_texts, data.train_labels)
    preds = predict(model, data.test_texts)
    matrix = confusion_matrix(data.test_labels, preds, model.classes)
    accuracy = float(np.mean(np.array(preds) == np.array(data.test_labels)))
    f1 = macro_f1(matrix)
    files = [
        plot_confusion(matrix, model.classes, output),
        plot_top_words(model, output),
        plot_metrics(accuracy, f1, output),
    ]
    pred_path = output / "nb_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "predicted_label"])
        writer.writerows(zip(data.test_texts, data.test_labels, preds))
    files.append(pred_path)
    report = {
        "project": "naive_bayes_text_classifier",
        "train_size": len(data.train_texts),
        "test_size": len(data.test_texts),
        "class_count": len(model.classes),
        "vocabulary_size": len(model.vocabulary),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(f1, 6),
        "generated_files": [p.name for p in files],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pr_assets/naive_bayes_text_classifier"),
        help="Directory for generated figures and metrics.",
    )
    parser.add_argument("--seed", type=int, default=917)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
