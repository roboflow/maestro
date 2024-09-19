import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def run_predictions(
    loader: DataLoader, processor: AutoProcessor, model: AutoModelForCausalLM
) -> tuple[list[str], list[str], list[str], list[Image.Image]]:
    questions_total = []
    expected_answers_total = []
    generated_answers_total = []
    images_total = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="running predictions", unit="batch")
        for inputs, questions, answers, images in progress_bar:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
            generated_answers = processor.batch_decode(generated_ids, skip_special_tokens=False)

            questions_total.extend(questions)
            expected_answers_total.extend(answers)
            generated_answers_total.extend(generated_answers)
            images_total.extend(images)

    return (questions_total, expected_answers_total, generated_answers_total, images_total)
