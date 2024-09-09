from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import matplotlib.pyplot as plt


class BaseMetric(ABC):
    """
    Abstract base class for custom metrics. Subclasses must implement
    the 'describe' and 'compute' methods.
    """

    @abstractmethod
    def describe(self) -> List[str]:
        """
        Describe the names of the metrics that this class will compute.

        Returns:
            List[str]: A list of metric names that will be computed.
        """
        pass

    @abstractmethod
    def compute(self, targets: List[Any], predictions: List[Any]) -> Dict[str, float]:
        """
        Compute the metric based on the targets and predictions.

        Args:
            targets (List[Any]): The ground truth.
            predictions (List[Any]): The prediction result.

        Returns:
            Dict[str, float]: A dictionary of computed metrics with metric names as
                keys and their values.
        """
        pass


class MetricsTracker:

    @classmethod
    def init(cls, metrics: List[str]) -> MetricsTracker:
        return cls(metrics={metric: [] for metric in metrics})

    def __init__(self, metrics: Dict[str, List[Tuple[int, int, float]]]):
        self._metrics = metrics

    def register(self, metric: str, epoch: int, step: int, value: float) -> None:
        self._metrics[metric].append((epoch, step, value))

    def describe_metrics(self) -> List[str]:
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
        self,
        output_dir: str = None,
        filename: str = None
    ) -> Dict[str, List[Dict[str, float]]]:
        metrics_data = {}
        for metric, values in self._metrics.items():
            metrics_data[metric] = [
                {'epoch': epoch, 'step': step, 'value': value}
                for epoch, step, value
                in values
            ]

        if output_dir and filename:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as file:
                json.dump(metrics_data, file, indent=4)

        return metrics_data


def aggregate_by_epoch(metric_values: List[Tuple[int, int, float]]) -> Dict[int, float]:
    """
    Aggregates metric values by epoch, calculating the average for each epoch.

    Args:
        metric_values (List[Tuple[int, int, float]]): A list of tuples containing
            (epoch, step, value) for each metric measurement.

    Returns:
        Dict[int, float]: A dictionary with epochs as keys and average metric values as values.
    """
    epoch_data = defaultdict(list)
    for epoch, step, value in metric_values:
        epoch_data[epoch].append(value)
    avg_per_epoch = {
        epoch: sum(values) / len(values)
        for epoch, values
        in epoch_data.items()
    }
    return avg_per_epoch


def save_metric_plots(
    training_tracker: MetricsTracker,
    validation_tracker: MetricsTracker,
    output_dir: str
):
    """
    Saves plots of training and validation metrics over epochs.

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
            training_values = training_tracker.get_metric_values(
                metric=metric, with_index=True)
            training_avg_values = aggregate_by_epoch(training_values)
            training_epochs = sorted(training_avg_values.keys())
            training_vals = [training_avg_values[epoch] for epoch in training_epochs]
            plt.plot(
                training_epochs,
                training_vals,
                label=f'Training {metric}',
                marker='o',
                linestyle='-',
                color='blue'
            )

        if metric in validation_metrics:
            validation_values = validation_tracker.get_metric_values(
                metric=metric, with_index=True)
            validation_avg_values = aggregate_by_epoch(validation_values)
            validation_epochs = sorted(validation_avg_values.keys())
            validation_vals = [
                validation_avg_values[epoch]
                for epoch
                in validation_epochs
            ]
            plt.plot(
                validation_epochs,
                validation_vals,
                label=f'Validation {metric}',
                marker='o',
                linestyle='--',
                color='orange'
            )

        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{metric}_plot.png')
        plt.close()
