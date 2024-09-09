from __future__ import annotations

import base64
import html
import io
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
from tabulate import tabulate
import time


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


class MetricsDisplay:
    def __init__(self, trackers: Dict[str, MetricsTracker]):
        self.trackers = trackers
        self.is_ipython = self._check_ipython()

    def _check_ipython(self):
        try:
            import IPython
            if IPython.get_ipython() is not None:
                return True
        except ImportError:
            pass
        return False

    def display_metrics(self):
        data = {}
        all_metrics = set()

        for tracker_name, tracker in self.trackers.items():
            metrics = tracker.describe_metrics()
            for metric in metrics:
                all_metrics.add(f"{tracker_name}_{metric}")
                values = tracker.get_metric_values(metric, with_index=True)
                if values:
                    avg_values = aggregate_by_epoch(values)
                    for epoch, value in avg_values.items():
                        if epoch not in data:
                            data[epoch] = {"epoch": epoch}
                        data[epoch][f"{tracker_name}_{metric}"] = f"{value:.4f}"

        headers = ["epoch"] + sorted(list(all_metrics))
        table_data = []
        for epoch in sorted(data.keys()):
            row = [epoch]
            for header in headers[1:]:
                value = data[epoch].get(header, "-")
                row.append(value)
            table_data.append(row)

        if self.is_ipython:
            self._display_ipython(table_data, headers)
        else:
            self._display_terminal(table_data, headers)

    def _display_ipython(self, data, headers):
        from IPython.display import display, HTML, clear_output
        
        clear_output(wait=True)
        table = tabulate(data, headers=headers, tablefmt="html", stralign="center")
        styled_table = f"""
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #ffffff;
                color: #333;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f5f5f5;
                color: #000;
            }}
        </style>
        {table}
        """
        display(HTML(styled_table))

    def _display_terminal(self, data, headers):
        print("\033c", end="")  # Clear the terminal screen
        print(tabulate(data, headers=headers, tablefmt="pretty"))

    def update_display(self):
        self.display_metrics()
        if not self.is_ipython:
            time.sleep(0.5)  # Add a small delay for terminal display


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



def display_results(
    prompts: List[str],
    expected_responses: List[str],
    generated_texts: List[str],
    images: List[Image.Image]
) -> None:
    """
    Display the results of model inference in IPython environments.

    This function attempts to display the results (prompts, expected responses,
    generated texts, and images) in an HTML format if running in an IPython
    environment. If not in IPython or if there's an ImportError, it silently passes.

    Args:
        prompts (List[str]): List of input prompts.
        expected_responses (List[str]): List of expected responses.
        generated_texts (List[str]): List of texts generated by the model.
        images (List[Image.Image]): List of input images.

    Returns:
        None
    """
    try:
        import IPython
        if IPython.get_ipython() is not None:
            from IPython.display import display, HTML
            html_out = create_html_output(prompts, expected_responses, generated_texts, images)
            display(HTML(html_out))
    except ImportError:
        pass  # Skip visualization if required libraries are not available


def create_html_output(
    prompts: List[str],
    expected_responses: List[str],
    generated_texts: List[str],
    images: List[Image.Image]
) -> str:
    """
    Create an HTML string to display the results of model inference.

    This function generates an HTML string that includes styled divs for each
    result, containing the input image, prompt, expected response, and generated text.

    Args:
        prompts (List[str]): List of input prompts.
        expected_responses (List[str]): List of expected responses.
        generated_texts (List[str]): List of texts generated by the model.
        images (List[Image.Image]): List of input images.

    Returns:
        str: An HTML string containing the formatted results.
    """
    html_out = "<style>.result-container{display:flex;margin-bottom:20px;border:1px solid #ddd;padding:10px;}.image-container{flex:0 0 256px;}.text-container{flex:1;margin-left:20px;}.prompt,.expected,.generated{margin-bottom:10px;}</style>"
    count = min(8, len(images))  # Display up to 8 examples
    for i in range(count):
        html_out += f"""
        <div class="result-container">
            <div class="image-container">
                <img style="width:256px; height:256px;" src="{render_inline(images[i])}" />
            </div>
            <div class="text-container">
                <div class="prompt"><strong>Prompt:</strong> {html.escape(prompts[i])}</div>
                <div class="expected"><strong>Expected:</strong> {html.escape(expected_responses[i])}</div>
                <div class="generated"><strong>Generated:</strong> {html.escape(generated_texts[i])}</div>
            </div>
        </div>
        """
    return html_out


def render_inline(image: Image.Image, resize: Tuple[int, int] = (256, 256)) -> str:
    """
    Convert an image into an inline HTML string.

    This function takes an image, resizes it, and converts it to a base64-encoded
    string that can be used as the source for an HTML img tag.

    Args:
        image (Image.Image): The input image to be converted.
        resize (Tuple[int, int], optional): The dimensions to resize the image to.
            Defaults to (256, 256).

    Returns:
        str: A string containing the data URI for the image, ready to be used
        in an HTML img tag's src attribute.
    """
    image = image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format='jpeg')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_b64}"