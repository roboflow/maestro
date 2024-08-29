import os
import random
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import supervision as sv
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from maestro.trainer.common.data_loaders.datasets import DetectionDataset
from maestro.trainer.common.utils.file_system import save_json
from maestro.trainer.common.utils.metrics_tracing import MetricsTracker
from maestro.trainer.models.florence_2.data_loading import prepare_detection_dataset


DETECTION_CLASS_PATTERN = r"([a-zA-Z0-9 ]+ of [a-zA-Z0-9 ]+)<loc_\d+>"


def prepare_detection_training_summary(
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    dataset_location: str,
    split_name: str,
    training_dir: str,
    num_samples_to_visualise: int,
    device: torch.device,
) -> None:
    dataset = prepare_detection_dataset(
        dataset_location=dataset_location,
        split_name=split_name,
    )
    if dataset is None:
        return None
    targets, predictions, post_processed_text_outputs = get_ground_truths_and_predictions(
        dataset=dataset,
        processor=processor,
        model=model,
        split_name=split_name,
        device=device,
    )
    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )
    print(f"{split_name} | map50_95: {mean_average_precision.map50_95:.2f}")
    print(f"{split_name} | map50: {mean_average_precision.map50:.2f}")
    print(f"{split_name} | map75: {mean_average_precision.map75:.2f}")
    dump_metrics(
        training_dir=training_dir,
        split_name=split_name,
        metrics=mean_average_precision,
    )
    dump_post_processed_outputs(
        dataset=dataset,
        post_processed_text_outputs=post_processed_text_outputs,
        training_dir=training_dir,
        split_name=split_name,
    )
    dump_visualised_samples(
        dataset=dataset,
        targets=targets,
        predictions=predictions,
        num_samples_to_visualise=num_samples_to_visualise,
        training_dir=training_dir,
        split_name=split_name,
    )


def get_ground_truths_and_predictions(
    dataset: DetectionDataset,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    split_name: str,
    device: torch.device,
) -> Tuple[List[sv.Detections], List[sv.Detections], List[str]]:
    classes = extract_classes(dataset=dataset)
    targets = []
    predictions = []
    post_processed_text_outputs = []
    for idx in tqdm(list(range(len(dataset))), desc=f"Generating {split_name} predictions..."):
        image, data = dataset.dataset[idx]
        prefix = data["prefix"]
        suffix = data["suffix"]

        inputs = processor(text=prefix, images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        prediction = processor.post_process_generation(generated_text, task="<OD>", image_size=image.size)
        post_processed_text_outputs.append(prediction)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
        prediction = prediction[np.isin(prediction["class_name"], classes)]
        prediction.class_id = np.array([class_name.index(class_name) for class_name in prediction["class_name"]])
        prediction.confidence = np.ones(len(prediction))
        target = processor.post_process_generation(suffix, task="<OD>", image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        target.class_id = np.array([class_name.index(class_name) for class_name in target["class_name"]])
        targets.append(target)
        predictions.append(prediction)
    return targets, predictions, post_processed_text_outputs


def extract_classes(dataset: DetectionDataset) -> List[str]:
    class_set = set()
    for i in range(len(dataset.dataset)):
        image, data = dataset.dataset[i]
        suffix = data["suffix"]
        classes = re.findall(DETECTION_CLASS_PATTERN, suffix)
        class_set.update(classes)
    return sorted(class_set)


def dump_metrics(
    training_dir: str,
    split_name: str,
    metrics: sv.MeanAveragePrecision,
) -> None:
    target_path = os.path.join(training_dir, "metrics", split_name, f"metrics_{split_name}.json")
    content = {
        "map50_95": metrics.map50_95,
        "map50": metrics.map50,
        "map75": metrics.map75,
    }
    save_json(path=target_path, content=content)


def dump_post_processed_outputs(
    dataset: DetectionDataset,
    post_processed_text_outputs: List[str],
    training_dir: str,
    split_name: str,
) -> None:
    result_dict = {
        dataset.dataset.entries[idx]["image"]: output for idx, output in enumerate(post_processed_text_outputs)
    }
    target_path = os.path.join(
        training_dir,
        "model_debug",
        split_name,
        f"post_processed_text_predictions_{split_name}.json",
    )
    save_json(path=target_path, content=result_dict)


def dump_visualised_samples(
    dataset: DetectionDataset,
    targets: List[sv.Detections],
    predictions: List[sv.Detections],
    num_samples_to_visualise: int,
    training_dir: str,
    split_name: str,
) -> None:
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    selected_indices = all_indices[:num_samples_to_visualise]
    target_dir = os.path.join(training_dir, "predictions_visualised", split_name)
    os.makedirs(target_dir, exist_ok=True)
    boxes_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    for idx in tqdm(selected_indices, desc="Preparing predictions visualisations..."):
        image_name = dataset.dataset.entries[idx]["image"]
        image = dataset.dataset[idx][0]
        prediction_image = boxes_annotator.annotate(
            image.copy(),
            predictions[idx],
        )
        prediction_image = np.asarray(
            label_annotator.annotate(
                prediction_image,
                predictions[idx],
            )
        )[:, :, ::-1]
        target_image = boxes_annotator.annotate(
            image.copy(),
            targets[idx],
        )
        target_image = np.asarray(
            label_annotator.annotate(
                target_image,
                targets[idx],
            )
        )[:, :, ::-1]
        concatenated = cv2.hconcat([target_image, prediction_image])
        target_image_path = os.path.join(target_dir, image_name)
        cv2.imwrite(target_image_path, concatenated)


def summarise_training_metrics(
    training_metrics_tracker: MetricsTracker,
    validation_metrics_tracker: MetricsTracker,
    training_dir: str,
) -> None:
    summarise_metrics(metrics_tracker=training_metrics_tracker, training_dir=training_dir, split_name="train")
    summarise_metrics(metrics_tracker=validation_metrics_tracker, training_dir=training_dir, split_name="valid")


def summarise_metrics(
    metrics_tracker: MetricsTracker,
    training_dir: str,
    split_name: str,
) -> None:
    plots_dir_path = os.path.join(training_dir, "metrics", split_name)
    os.makedirs(plots_dir_path, exist_ok=True)
    for metric_name in metrics_tracker.describe_metrics():
        plot_path = os.path.join(plots_dir_path, f"metric_{metric_name}_plot.png")
        plt.clf()
        metric_values_with_index = metrics_tracker.get_metric_values(
            metric=metric_name,
            with_index=True,
        )
        xs = np.arange(0, len(metric_values_with_index))
        xticks_xs, xticks_labels = [], []
        previous = None
        for v, x in zip(metric_values_with_index, xs):
            if v[0] != previous:
                xticks_xs.append(x)
                xticks_labels.append(v[0])
            previous = v[0]
        ys = [e[2] for e in metric_values_with_index]
        plt.scatter(xs, ys, marker="x")
        plt.plot(xs, ys, linestyle="dashed", linewidth=0.3)
        plt.title(f"Value of {metric_name} for {split_name} set")
        plt.xticks(xticks_xs, labels=xticks_labels)
        plt.xlabel("Epochs")
        plt.savefig(plot_path, dpi=120)
        plt.clf()
