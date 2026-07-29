"""Dependency-free scalar reference for the counterfactual margin loss."""

import math


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return numerator / (left_norm * right_norm)


def counterfactual_margin_loss_reference(
    predictions: list[list[float]],
    positives: list[list[float]],
    negatives: list[list[float]],
    margin: float = 0.1,
) -> float:
    """Return the mean hinge loss for small CPU-only validation inputs."""
    if not (len(predictions) == len(positives) == len(negatives)):
        raise ValueError("prediction, positive, and negative batches must match")
    losses = [
        max(0.0, margin - _cosine(prediction, positive) + _cosine(prediction, negative))
        for prediction, positive, negative in zip(predictions, positives, negatives)
    ]
    return sum(losses) / len(losses)
