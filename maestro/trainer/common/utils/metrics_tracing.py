from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt


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


def aggregate_by_epoch(metric_values: List[Tuple[int, int, float]]) -> Dict[int, float]:
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
                x=training_epochs,
                y=training_vals,
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
                x=validation_epochs,
                y=validation_vals,
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
