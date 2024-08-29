import os
from typing import Optional, Tuple, Literal, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv

from maestro.trainer.common.configuration.env import CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE

DEFAULT_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_FLORENCE2_MODEL_REVISION = "refs/pr/20"
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else os.getenv(CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE)


def load_model(
    model_id: str = DEFAULT_FLORENCE2_MODEL_ID,
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: torch.device = DEVICE,
    cache_dir: Optional[str] = None,
) -> Tuple[AutoProcessor, AutoModelForCausalLM]:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, cache_dir=cache_dir
    ).to(device)
    return processor, model


def caption_image(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    task: Literal["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<VQA>"],
    prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    num_beams: int = 3,
) -> str:
    prompt = _pre_process_prompt(image=image, task=task, prompt=prompt)
    model_device = model.device
    inputs = processor(text=task, images=image, return_tensors="pt").to(model_device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
    return response[task]


TASKS_THAT_REQUIRE_PROMPT = {
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
    "<VQA>",
}


def segment_objects(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    task: Literal[
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "<REGION_TO_SEGMENTATION>",
    ],
    prompt: Optional[Union[str, tuple, list, np.ndarray]] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    num_beams: int = 3,
) -> sv.Detections:
    return _prompt_and_retrieve_detections(
        image=image,
        processor=processor,
        model=model,
        task=task,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
    )


def detect_objects(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    task: Literal[
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
        "<OCR_WITH_REGION>",
        "<OPEN_VOCABULARY_DETECTION>",
        "<REGION_TO_CATEGORY>",
        "<REGION_TO_DESCRIPTION>",
    ],
    prompt: Optional[Union[str, tuple, list, np.ndarray]] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    num_beams: int = 3,
) -> sv.Detections:
    return _prompt_and_retrieve_detections(
        image=image,
        processor=processor,
        model=model,
        task=task,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
    )


def _prompt_and_retrieve_detections(
    image: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    task: Literal[
        "<OD>",
        "<DENSE_REGION_CAPTION>",
        "<REGION_PROPOSAL>",
        "<OCR_WITH_REGION>",
        "<OPEN_VOCABULARY_DETECTION>",
        "<REGION_TO_CATEGORY>",
        "<REGION_TO_DESCRIPTION>",
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        "<REGION_TO_SEGMENTATION>",
    ],
    prompt: Optional[Union[str, tuple, list, np.ndarray]] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    num_beams: int = 3,
) -> sv.Detections:
    prompt = _pre_process_prompt(image=image, task=task, prompt=prompt)
    model_device = model.device
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model_device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )
    return sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2,
        result=response,
        resolution_wh=image.size,
    )


def _pre_process_prompt(
    image: Image.Image,
    task: str,
    prompt: Optional[Union[str, tuple, list, np.ndarray]] = None,
) -> str:
    if prompt is None:
        if task in TASKS_THAT_REQUIRE_PROMPT:
            raise ValueError(f"Task {task} requires prompt")
        return task
    if isinstance(prompt, tuple) or isinstance(prompt, list) or isinstance(prompt, np.ndarray):
        if len(prompt) != 4:
            raise ValueError("Expected sequence of 4 elements describing (x_min, y_min, x_max, y_max)")
        x_min, y_min, x_max, y_max = prompt
        x_min, x_max = round((x_min / image.width) * 1000), round((x_max / image.width) * 1000)
        y_min, y_max = round((y_min / image.height) * 1000), round((y_max / image.height) * 1000)
        return f"{task} <loc_{x_min}><loc_{y_min}><loc_{x_max}><loc_{y_max}>"
    return f"{task} {prompt}"
