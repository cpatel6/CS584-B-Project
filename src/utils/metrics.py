import time
from typing import Any, Callable, Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def measure_training_time(train_fn: Callable[..., Any], *args, **kwargs) -> float:
    start = time.perf_counter()
    train_fn(*args, **kwargs)
    return float(time.perf_counter() - start)


def measure_inference_speed(
    predict_fn: Callable[..., Any], inputs, repeats: int = 1
) -> Dict[str, float]:
    if repeats < 1:
        repeats = 1
    start = time.perf_counter()
    num_predictions = 0
    # Repeat predictions to smooth out noisy one-off timing variance.
    for _ in range(repeats):
        preds = predict_fn(inputs)
        num_predictions = len(preds)
    elapsed = max(time.perf_counter() - start, 1e-12)
    samples = len(inputs) * repeats
    return {
        "inference_time_sec": float(elapsed),
        "inference_samples_per_sec": float(samples / elapsed),
        "num_predictions": int(num_predictions),
    }
