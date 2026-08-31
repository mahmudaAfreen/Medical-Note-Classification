"""Metrics for single-label medical span classification."""

from __future__ import annotations

from collections import Counter
from typing import Any


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_report(
    y_true: list[str],
    y_pred: list[str | None],
    labels: list[str],
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    total = len(y_true)
    valid_pred = [pred if pred in labels else None for pred in y_pred]
    correct = sum(truth == pred for truth, pred in zip(y_true, valid_pred))
    invalid = sum(pred is None for pred in valid_pred)

    per_label = {}
    true_counts = Counter(y_true)
    pred_counts = Counter(pred for pred in valid_pred if pred is not None)
    for label in labels:
        tp = sum(
            truth == label and pred == label
            for truth, pred in zip(y_true, valid_pred)
        )
        fp = pred_counts[label] - tp
        fn = true_counts[label] - tp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_counts[label],
            "predicted": pred_counts[label],
        }

    macro_f1 = safe_divide(sum(row["f1"] for row in per_label.values()), len(labels))
    weighted_f1 = safe_divide(
        sum(row["f1"] * row["support"] for row in per_label.values()),
        total,
    )

    return {
        "accuracy": safe_divide(correct, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": safe_divide(correct, total),
        "invalid_predictions": invalid,
        "total": total,
        "per_label": per_label,
    }


def section_reports(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    reports = {}
    sections = sorted({row.get("section", "") for row in rows})
    for section in sections:
        section_rows = [row for row in rows if row.get("section", "") == section]
        reports[section] = classification_report(
            [row["label"] for row in section_rows],
            [row.get("predicted_label") for row in section_rows],
            labels,
        )
    return reports
