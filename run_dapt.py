#!/usr/bin/env python3
"""
Run SSL DAPT + LoRA Training with YAML Configuration

Usage:
    python run_dapt.py --config configs/qwen3_4b_dapt.yaml
    python run_dapt.py --config configs/qwen3_8b_dapt.yaml --data_path ./my_data
"""

import argparse
import logging
import yaml
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from data_utils import load_json_corpus, PackedDataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_bnb_config(model_config: dict):
    """Create BitsAndBytes quantization config."""
    if model_config.get("use_4bit", False):
        compute_dtype = getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.get("use_nested_quant", False),
        )
    elif model_config.get("use_8bit", False):
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def create_lora_config(lora_config: dict) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(config: dict):
    """Load model and tokenizer."""
    model_config = config["model"]
    lora_cfg = config["lora"]

    model_name = model_config["name"]
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create quantization config
    bnb_config = create_bnb_config(model_config)

    # Load model
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Try to use Flash Attention 2
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except Exception:
        logger.warning("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare for k-bit training if quantized
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        )

    # Apply LoRA
    lora_config = create_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run SSL DAPT + LoRA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path from config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if args.data_path:
        config["data"]["train_path"] = args.data_path
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.use_wandb:
        config["logging"]["use_wandb"] = True

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Load data
    data_config = config["data"]
    logger.info(f"Loading data from: {data_config['train_path']}")

    texts = load_json_corpus(
        data_config["train_path"],
        text_field=data_config.get("text_field", "text"),
    )

    # Create dataset
    train_dataset = PackedDataset(
        texts,
        tokenizer,
        max_seq_length=data_config.get("max_seq_length", 2048),
        pack_sequences=data_config.get("pack_sequences", True),
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    train_config = config["training"]
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config.get("num_epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        learning_rate=train_config.get("learning_rate", 2e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler", "cosine"),
        logging_steps=train_config.get("logging_steps", 10),
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 3),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        optim=train_config.get("optim", "paged_adamw_32bit"),
        max_grad_norm=train_config.get("max_grad_norm", 0.3),
        seed=train_config.get("seed", 42),
        report_to="wandb" if config["logging"].get("use_wandb", False) else "none",
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    output_dir = train_config["output_dir"]
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save LoRA adapter
    lora_dir = Path(output_dir) / "lora_adapter"
    model.save_pretrained(lora_dir)
    logger.info(f"LoRA adapter saved to {lora_dir}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
