#!/usr/bin/env python3
"""
Run SSL DAPT + LoRA Training with YAML Configuration

Domain-Adaptive Pre-Training (DAPT) optimized for knowledge injection.

Usage:
    # Full precision training (recommended for best quality)
    python run_dapt.py --config configs/qwen3_4b_dapt.yaml

    # With quantization for limited GPU memory
    python run_dapt.py --config configs/qwen3_8b_dapt.yaml --use_quantization

    # Override data path
    python run_dapt.py --config configs/qwen3_4b_dapt.yaml --data_path ./my_corpus
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional, Any

import yaml
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
from torch.utils.data import Dataset
import json

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


class DAPTCorpusDataset(Dataset):
    """Dataset for DAPT with sequence packing."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        text_field: str = "text",
        pack_sequences: bool = True,
        min_sequence_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_field = text_field
        self.examples = []

        texts = self._load_texts(data_path)
        logger.info(f"Loaded {len(texts)} documents")

        if pack_sequences:
            self._pack_texts(texts, min_sequence_length)
        else:
            self._tokenize_texts(texts, min_sequence_length)

        logger.info(f"Created {len(self.examples)} training examples")

    def _load_texts(self, data_path: str) -> List[str]:
        data_path = Path(data_path)
        texts = []

        def process_item(item: Any) -> Optional[str]:
            if isinstance(item, str):
                return item.strip() if item.strip() else None
            if isinstance(item, dict):
                if self.text_field in item:
                    return str(item[self.text_field]).strip()
                for field in ["text", "content", "body", "document"]:
                    if field in item:
                        return str(item[field]).strip()
                strings = [str(v) for v in item.values() if isinstance(v, str)]
                return " ".join(strings).strip() if strings else None
            return None

        def load_file(file_path: Path):
            logger.info(f"Loading {file_path}")
            try:
                if file_path.suffix == ".jsonl":
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    text = process_item(json.loads(line))
                                    if text and len(text) > 10:
                                        texts.append(text)
                                except json.JSONDecodeError:
                                    continue
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            text = process_item(item)
                            if text and len(text) > 10:
                                texts.append(text)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        if data_path.is_file():
            load_file(data_path)
        elif data_path.is_dir():
            for pattern in ["**/*.json", "**/*.jsonl"]:
                for fp in sorted(data_path.glob(pattern)):
                    load_file(fp)

        return texts

    def _tokenize_texts(self, texts: List[str], min_len: int):
        for text in texts:
            enc = self.tokenizer(
                text, truncation=True, max_length=self.max_seq_length,
                padding=False, return_tensors=None
            )
            if len(enc["input_ids"]) >= min_len:
                self.examples.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]
                })

    def _pack_texts(self, texts: List[str], min_len: int):
        random.shuffle(texts)
        all_tokens = []
        eos_id = self.tokenizer.eos_token_id

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(eos_id)

        for i in range(0, len(all_tokens), self.max_seq_length):
            chunk = all_tokens[i:i + self.max_seq_length]
            if len(chunk) >= min_len:
                self.examples.append({
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_quantization_config(model_config: dict) -> Optional[BitsAndBytesConfig]:
    """Create quantization config if enabled."""
    if not model_config.get("use_quantization", False):
        return None

    bits = model_config.get("quantization_bits", 4)
    if bits == 4:
        compute_dtype = getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.get("use_nested_quant", False),
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def create_lora_config(lora_config: dict) -> LoraConfig:
    """Create LoRA configuration for DAPT."""
    return LoraConfig(
        r=lora_config.get("r", 128),
        lora_alpha=lora_config.get("alpha", 256),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(config: dict, use_quantization: bool = False):
    """Load model and tokenizer for DAPT."""
    model_config = config["model"]
    lora_cfg = config["lora"]

    model_name = model_config["name"]
    logger.info(f"Loading model: {model_name}")

    # Warn if using instruct model
    if "Instruct" in model_name or "Chat" in model_name:
        logger.warning("Using instruct model for DAPT. Base models recommended.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Override quantization setting
    model_config["use_quantization"] = use_quantization
    bnb_config = create_quantization_config(model_config)

    # Model kwargs
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        logger.info(f"Using {model_config.get('quantization_bits', 4)}-bit quantization")
    else:
        dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))
        model_kwargs["torch_dtype"] = dtype
        logger.info(f"Using full precision ({model_config.get('torch_dtype', 'bfloat16')})")

    # Try Flash Attention 2
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare for training
    gradient_checkpointing = config["training"].get("gradient_checkpointing", True)
    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = create_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run DAPT with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--use_quantization", action="store_true", help="Enable quantization")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.data_path:
        config["data"]["train_path"] = args.data_path
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.use_wandb:
        config["logging"]["use_wandb"] = True

    # Load model
    model, tokenizer = load_model_and_tokenizer(config, use_quantization=args.use_quantization)

    # Load data
    data_config = config["data"]
    train_dataset = DAPTCorpusDataset(
        data_path=data_config["train_path"],
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 2048),
        text_field=data_config.get("text_field", "text"),
        pack_sequences=data_config.get("pack_sequences", True),
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments (DAPT optimized)
    train_config = config["training"]
    optim = "paged_adamw_32bit" if args.use_quantization else train_config.get("optim", "adamw_torch")

    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config.get("num_epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        learning_rate=train_config.get("learning_rate", 5e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_config.get("lr_scheduler", "cosine"),
        logging_steps=train_config.get("logging_steps", 10),
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 3),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        optim=optim,
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        seed=train_config.get("seed", 42),
        report_to="wandb" if config["logging"].get("use_wandb", False) else "none",
        remove_unused_columns=False,
        save_safetensors=True,
    )

    # Log config
    logger.info("=" * 60)
    logger.info("DAPT Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Quantization: {'Enabled' if args.use_quantization else 'Disabled'}")
    logger.info(f"LoRA rank: {config['lora'].get('r', 128)}")
    logger.info(f"Learning rate: {train_config.get('learning_rate', 5e-5)}")
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info("=" * 60)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting DAPT training...")
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

    logger.info("DAPT training complete!")


if __name__ == "__main__":
    main()
