import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from maestro.trainer.models.qwen_2_5_vl.checkpoints import DEVICE
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation
from qwen_vl_utils import process_vision_info


def predict_with_inputs(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device = DEVICE,
    max_new_tokens: int = 1024
) -> list[str]:
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            pixel_values=pixel_values.to(device),
            image_grid_thw=image_grid_thw.to(device),
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            generated_sequence[len(input_sequence):]
            for input_sequence, generated_sequence
            in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text


def predict(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    system_message: str | None = None,
    device: torch.device = DEVICE,
    max_new_tokens: int = 1024
) -> str:
    conversation = format_conversation(image=image, prefix=prefix, system_message=system_message)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )
    return predict_with_inputs(
        **inputs,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens
    )[0]
