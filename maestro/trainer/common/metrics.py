from __future__ import annotations

import base64
import html
import io
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import supervision as sv
from evaluate import load
from jiwer import cer, wer
from PIL import Image
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class BaseMetric(ABC):
    """Abstract base class for custom metrics. Subclasses must implement
    the 'describe' and 'compute' methods.
    """

    @abstractmethod
    def describe(self) -> list[str]:
        """Describe the names of the metrics that this class will compute.

        Returns:
            List[str]: A list of metric names that will be computed.
        """
        pass

    @abstractmethod
    def compute(self, targets: list[Any], predictions: list[Any]) -> dict[str, float]:
        """Compute the metric based on the targets and predictions.

        Args:
            targets (List[Any]): The ground truth.
            predictions (List[Any]): The prediction result.

        Returns:
            Dict[str, float]: A dictionary of computed metrics with metric names as
                keys and their values.
        """
        pass


class MeanAveragePrecisionMetric(BaseMetric):
    """A class used to compute the Mean Average Precision (mAP) metric.

    mAP is a popular metric for object detection tasks, measuring the average precision
    across all classes and IoU thresholds.
    """

    name = "mean_average_precision"

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        """
        return ["map50:95", "map50", "map75"]

    def compute(self, targets: list[sv.Detections], predictions: list[sv.Detections]) -> dict[str, float]:
        """Computes the mAP metrics based on the targets and predictions.

        Args:
            targets (List[sv.Detections]): The ground truth detections.
            predictions (List[sv.Detections]): The predicted detections.

        Returns:
            Dict[str, float]: A dictionary of computed mAP metrics with metric names as
                keys and their values.
        """
        result = MeanAveragePrecision().update(targets=targets, predictions=predictions).compute()
        return {"map50:95": result.map50_95, "map50": result.map50, "map75": result.map75}


class WordErrorRateMetric(BaseMetric):
    """A class used to compute the Word Error Rate (WER) metric.

    WER measures the edit distance between predicted and reference transcriptions
    at the word level, commonly used in speech recognition and machine translation.
    """

    name = "word_error_rate"

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        """
        return ["wer"]

    def compute(self, targets: list[str], predictions: list[str]) -> dict[str, float]:
        """Computes the WER metric based on the targets and predictions.

        Args:
            targets (List[str]): The ground truth texts (references), where each element
                represents the reference text for the corresponding prediction.
            predictions (List[str]): The predicted texts (hypotheses) to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the computed WER score, with the
                metric name ("wer") as the key and its value as the score.
        """
        if len(targets) != len(predictions):
            raise ValueError("The number of targets and predictions must be the same.")

        total_wer = 0.0
        count = len(targets)

        for target, prediction in zip(targets, predictions):
            total_wer += wer(target, prediction)

        average_wer = total_wer / count if count > 0 else 0.0
        return {"wer": average_wer}


class CharacterErrorRateMetric(BaseMetric):
    """A class used to compute the Character Error Rate (CER) metric.

    CER is similar to WER but operates at the character level, making it useful for
    tasks like optical character recognition (OCR) and handwriting recognition.
    """

    name = "character_error_rate"

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        """
        return ["cer"]

    def compute(self, targets: list[str], predictions: list[str]) -> dict[str, float]:
        """Computes the CER metric based on the targets and predictions.

        Args:
            targets (List[str]): The ground truth texts (references), where each element
                represents the reference text for the corresponding prediction.
            predictions (List[str]): The predicted texts (hypotheses) to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the computed CER score, with the
                metric name ("cer") as the key and its value as the score.
        """
        if len(targets) != len(predictions):
            raise ValueError("The number of targets and predictions must be the same.")

        total_cer = 0.0
        count = len(targets)

        for target, prediction in zip(targets, predictions):
            total_cer += cer(target, prediction)

        average_cer = total_cer / count if count > 0 else 0.0
        return {"cer": average_cer}


class TranslationErrorRateMetric(BaseMetric):
    """A class used to compute the Translation Error Rate (TER) metric.

    TER measures the minimum number of edits (insertions, deletions, substitutions, and shifts)
    needed to transform a predicted text into its reference text, making it useful for
    evaluating machine translation and other text generation tasks.
    """

    name = "translation_error_rate"

    def __init__(self, case_sensitive: bool = True):
        """Initialize the TER metric.

        Args:
            case_sensitive (bool, optional): Whether to perform case-sensitive comparison.
                Defaults to True.
        """
        self.ter = load("ter")
        self.case_sensitive = case_sensitive

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        """
        return ["ter"]

    def compute(self, targets: list[str], predictions: list[str]) -> dict[str, float]:
        """Computes the TER metric based on the targets and predictions.

        Args:
            targets (List[str]): The ground truth texts (references), where each element
                represents the reference text for the corresponding prediction.
            predictions (List[str]): The predicted texts (hypotheses) to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the computed TER score, with the
                metric name ("ter") as the key and its value as the score.
        """
        if len(targets) != len(predictions):
            raise ValueError("The number of targets and predictions must be the same.")

        results = self.ter.compute(predictions=predictions, references=targets, case_sensitive=self.case_sensitive)
        return {"ter": results["score"]}


class BLEUMetric(BaseMetric):
    """A class used to compute the BLEU (Bilingual Evaluation Understudy) metric.

    BLEU is a popular metric for evaluating the quality of text predictions in natural
    language processing tasks, particularly machine translation. It measures the
    similarity between the predicted text and one or more reference texts based on
    n-gram precision, brevity penalty, and other factors.
    """

    bleu = load("bleu")
    name = "bleu"

    def describe(self) -> list[str]:
        """Returns a list of metric names that this class will compute.

        Returns:
            List[str]: A list of metric names.
        """
        return ["bleu"]

    def compute(self, targets: list[str], predictions: list[str]) -> dict[str, float]:
        """Computes the BLEU metric based on the targets and predictions.

        Args:
            targets (List[str]): The ground truth texts (references), where each element
                represents the reference text for the corresponding prediction.
            predictions (List[str]): The predicted texts (hypotheses) to be evaluated.

        Returns:
            Dict[str, float]: A dictionary containing the computed BLEU score, with the
                metric name ("bleu") as the key and its value as the score.
        """
        if len(targets) != len(predictions):
            raise ValueError("The number of targets and predictions must be the same.")

        results = self.bleu.compute(predictions=predictions, references=targets)
        return {"bleu": results["bleu"]}


class MetricsTracker:
    @classmethod
    def init(cls, metrics: list[str]) -> MetricsTracker:
        return cls(metrics={metric: [] for metric in metrics})

    def __init__(self, metrics: dict[str, list[tuple[int, int, float]]]) -> None:
        self._metrics = metrics

    def register(self, metric: str, epoch: int, step: int, value: float) -> None:
        self._metrics[metric].append((epoch, step, value))

    def describe_metrics(self) -> list[str]:
        return list(self._metrics.keys())

    def get_metric_values(
        self,
        metric: str,
        with_index: bool = True,
    ) -> list:
        if with_index:
            return self._metrics[metric]
        return [value[2] for value in self._metrics[metric]]

    def as_json(
        self, output_dir: Optional[str] = None, filename: Optional[str] = None
    ) -> dict[str, list[dict[str, float]]]:
        metrics_data = {}
        for metric, values in self._metrics.items():
            metrics_data[metric] = [{"epoch": epoch, "step": step, "value": value} for epoch, step, value in values]

        if output_dir and filename:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as file:
                json.dump(metrics_data, file, indent=4)

        return metrics_data


def aggregate_by_epoch(metric_values: list[tuple[int, int, float]]) -> dict[int, float]:
    """Aggregates metric values by epoch, calculating the average for each epoch.

    Args:
        metric_values (List[Tuple[int, int, float]]): A list of tuples containing
            (epoch, step, value) for each metric measurement.

    Returns:
        Dict[int, float]: A dictionary with epochs as keys and average metric values as values.
    """
    epoch_data = defaultdict(list)
    for epoch, step, value in metric_values:
        epoch_data[epoch].append(value)
    avg_per_epoch = {epoch: sum(values) / len(values) for epoch, values in epoch_data.items()}
    return avg_per_epoch


def save_metric_plots(training_tracker: MetricsTracker, validation_tracker: MetricsTracker, output_dir: str) -> None:
    """Saves plots of training and validation metrics over epochs.

    Args:
        training_tracker (MetricsTracker): Tracker containing training metrics.
        validation_tracker (MetricsTracker): Tracker containing validation metrics.
        output_dir (str): Directory to save the generated plots.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_metrics = training_tracker.describe_metrics()
    validation_metrics = validation_tracker.describe_metrics()
    all_metrics = set(training_metrics + validation_metrics)

    for metric in all_metrics:
        plt.figure(figsize=(8, 6))

        if metric in training_metrics:
            training_values = training_tracker.get_metric_values(metric=metric, with_index=True)
            training_avg_values = aggregate_by_epoch(training_values)
            training_epochs = sorted(training_avg_values.keys())
            training_vals = [training_avg_values[epoch] for epoch in training_epochs]
            plt.plot(
                training_epochs, training_vals, label=f"Training {metric}", marker="o", linestyle="-", color="blue"
            )

        if metric in validation_metrics:
            validation_values = validation_tracker.get_metric_values(metric=metric, with_index=True)
            validation_avg_values = aggregate_by_epoch(validation_values)
            validation_epochs = sorted(validation_avg_values.keys())
            validation_vals = [validation_avg_values[epoch] for epoch in validation_epochs]
            plt.plot(
                validation_epochs,
                validation_vals,
                label=f"Validation {metric}",
                marker="o",
                linestyle="--",
                color="orange",
            )

        plt.title(f"{metric.capitalize()} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(f"{metric.capitalize()} Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{metric}_plot.png")
        plt.close()
