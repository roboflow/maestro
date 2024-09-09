import os
import random
import re
from typing import List, Dict
from typing import Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from maestro.trainer.common.data_loaders.datasets import DetectionDataset
from maestro.trainer.common.utils.file_system import save_json
from maestro.trainer.common.utils.metrics import BaseMetric
from maestro.trainer.models.florence_2.data_loading import prepare_detection_dataset

DETECTION_CLASS_PATTERN = r"([a-zA-Z0-9 ]+ of [a-zA-Z0-9 ]+)<loc_\d+>"


class MeanAveragePrecisionMetric(BaseMetric):

    def describe(self) -> List[str]:
        return ["map50:95", "map50", "map75"]

    def compute(
        self,
        targets: List[sv.Detections],
        predictions: List[sv.Detections]
    ) -> Dict[str, float]:
        result = MeanAveragePrecision().update(targets, predictions).compute()
        return {
            "map50:95": result.map50_95,
            "map50": result.map50,
            "map75": result.map75
        }


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
    device: torch.device,
) -> Tuple[List[sv.Detections], List[sv.Detections], List[str]]:
    classes = extract_classes(dataset=dataset)
    targets = []
    predictions = []
    post_processed_text_outputs = []
    for idx in tqdm(list(range(len(dataset))), desc="Generating predictions..."):
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
