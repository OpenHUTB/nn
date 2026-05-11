from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from nb_text_demo import confusion_matrix, fit_naive_bayes, make_dataset, predict, run


def test_naive_bayes_classifies_synthetic_text() -> None:
    data = make_dataset(seed=917)
    model = fit_naive_bayes(data.train_texts, data.train_labels)
    preds = predict(model, data.test_texts)
    accuracy = sum(a == b for a, b in zip(data.test_labels, preds)) / len(preds)
    assert accuracy > 0.9
    matrix = confusion_matrix(data.test_labels, preds, model.classes)
    assert matrix.shape == (3, 3)


def test_exports(tmp_path: Path) -> None:
    metrics = run(tmp_path)
    assert metrics["accuracy"] > 0.9
    assert metrics["macro_f1"] > 0.9
    assert (tmp_path / "nb_confusion_matrix.png").exists()
    assert (tmp_path / "nb_top_words.png").exists()
    assert (tmp_path / "nb_metric_summary.png").exists()
    assert (tmp_path / "nb_predictions.csv").exists()


if __name__ == "__main__":
    test_naive_bayes_classifies_synthetic_text()
    with tempfile.TemporaryDirectory() as tmp:
        test_exports(Path(tmp))
    print("naive_bayes_text_classifier tests passed")
