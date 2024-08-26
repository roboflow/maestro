from __future__ import annotations

from typing import Dict, Tuple, List


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
