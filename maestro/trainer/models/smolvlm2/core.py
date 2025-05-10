import os
from typing import Optional, Union

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer


class SmolVLM2Core:
    """Core SmolVLM2 model implementation."""

    def __init__(
        self,
        model_name: str = "smol-ai/smolvlm2-500m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Initialize SmolVLM2 model.

        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on
            **kwargs: Additional arguments to pass to the model
        """
        self.model_name = model_name
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        self.model.to(device)

    def process_inputs(self, images: Union[str, list[str]], prompt: Optional[str] = None) -> dict:
        """Process input images and text."""
        if isinstance(images, str):
            images = [images]

        return self.processor(images=images, text=prompt if prompt else "", return_tensors="pt").to(self.device)

    def generate(self, inputs: dict, max_new_tokens: int = 512, **kwargs) -> torch.Tensor:
        """Generate text from processed inputs."""
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)

    def decode_outputs(self, outputs: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        """Decode model outputs to text."""
        return self.processor.batch_decode(outputs, skip_special_tokens=skip_special_tokens)


def train(config: dict) -> dict:
    """
    Train SmolVLM2 model with provided configuration.

    Args:
        config: Dictionary containing training configuration
            - dataset: Path to dataset directory or file
            - epochs: Number of training epochs
            - batch_size: Training batch size
            - optimization_strategy: Strategy for optimization (qlora, lora, freeze_vision)
            - metrics: List of metrics to evaluate during training
            - output_dir: Directory to save trained model
    Returns:
        Dictionary containing training results and metrics
    """

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, TrainingArguments

    from maestro.trainer.common.datasets.core import create_data_loaders, resolve_dataset_path
    from maestro.trainer.models.smolvlm2.loaders import evaluation_collate_fn, train_collate_fn

    # Load dataset
    dataset_path = config["dataset"]
    dataset_location = resolve_dataset_path(dataset_path)
    if dataset_location is None:
        return {"error": "Dataset not found"}

    # Create model with the specified optimization strategy
    model_name = config.get("model_name", "smol-ai/smolvlm2-500m")
    strategy = config.get("optimization_strategy", "qlora")

    if strategy == "qlora":
        # Configure QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForVision2Seq.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    elif strategy == "lora":
        # Configure LoRA without quantization
        model = AutoModelForVision2Seq.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    elif strategy == "freeze_vision":
        # Freeze vision encoder, train only language model part
        model = AutoModelForVision2Seq.from_pretrained(model_name)

        # Freeze vision encoder parameters
        for param in model.vision_model.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Unsupported optimization strategy: {strategy}")

    # Load processor and datasets
    processor = AutoProcessor.from_pretrained(model_name)

    # Create processor wrapper to preprocess data before collating
    def process_batch(batch):
        processed_batch = []
        for item in batch:
            processed_item = processor(images=item.get("image"), text=item.get("text", ""), return_tensors="pt")
            processed_batch.append(processed_item)
        return processed_batch

    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=dataset_location,
        train_batch_size=config.get("batch_size", 4),
        train_collect_fn=lambda batch: train_collate_fn(process_batch(batch)),
        train_num_workers=config.get("num_workers", 0),
        test_batch_size=config.get("val_batch_size", config.get("batch_size", 4)),
        test_collect_fn=lambda batch: evaluation_collate_fn(process_batch(batch)),
        test_num_workers=config.get("val_num_workers", config.get("num_workers", 0)),
    )

    # Set up training arguments
    output_dir = config.get("output_dir", "./smolvlm2-finetuned")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 10),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("val_batch_size", config.get("batch_size", 4)),
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    # Safely handle potential None loaders by directly checking train_loader/valid_loader before accessing dataset attribute
    train_dataset = None
    if train_loader is not None:
        train_dataset = train_loader.dataset

    eval_dataset = None
    if valid_loader is not None:
        eval_dataset = valid_loader.dataset

    # Create data_collator that matches the train_collate_fn signature (doesn't pass processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: train_collate_fn(process_batch(batch)),
    )

    # Train model
    trainer.train()

    # Save model and processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Return results
    return {
        "model_path": output_dir,
        "metrics": trainer.state.log_history[-1] if trainer.state.log_history else {"loss": "N/A"},
        "status": "Training completed",
    }
