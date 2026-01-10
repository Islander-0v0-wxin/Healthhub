"""
SSL DAPT + LoRA Training Script for Qwen3 Models

This script implements Domain-Adaptive Pre-Training (DAPT) with LoRA
on Qwen3-4B or Qwen3-8B models using JSON/JSONL corpus data.

SSL (Self-Supervised Learning) DAPT continues pre-training a language model
on domain-specific text data using causal language modeling objective.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
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
from datasets import load_dataset
import wandb

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model and LoRA settings."""
    model_name: str = "Qwen/Qwen3-4B"  # or "Qwen/Qwen3-8B"
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str = "./data"
    max_seq_length: int = 2048
    text_field: str = "text"  # Field name in JSON containing text
    streaming: bool = False


@dataclass
class TrainConfig:
    """Configuration for training."""
    output_dir: str = "./outputs/qwen3_dapt_lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    seed: int = 42
    report_to: str = "wandb"  # or "none"


class JSONCorpusDataset(Dataset):
    """Dataset for loading JSON/JSONL corpus files."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        text_field: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_field = text_field
        self.examples = []

        self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def _load_data(self, data_path: str):
        """Load data from JSON or JSONL files."""
        data_path = Path(data_path)

        if data_path.is_file():
            self._load_file(data_path)
        elif data_path.is_dir():
            for file_path in data_path.glob("**/*.json"):
                self._load_file(file_path)
            for file_path in data_path.glob("**/*.jsonl"):
                self._load_file(file_path)
        else:
            raise ValueError(f"Data path {data_path} does not exist")

    def _load_file(self, file_path: Path):
        """Load a single JSON or JSONL file."""
        logger.info(f"Loading {file_path}")

        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._process_item(data)
        else:  # .json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        self._process_item(item)
                else:
                    self._process_item(data)

    def _process_item(self, item: Dict[str, Any]):
        """Process a single data item and extract text."""
        if isinstance(item, dict):
            # Try to get text from specified field or common fields
            text = None
            if self.text_field in item:
                text = item[self.text_field]
            elif "content" in item:
                text = item["content"]
            elif "body" in item:
                text = item["body"]
            elif "message" in item:
                text = item["message"]
            else:
                # Concatenate all string values
                text = " ".join(str(v) for v in item.values() if isinstance(v, str))

            if text and isinstance(text, str) and len(text.strip()) > 0:
                self.examples.append(text.strip())
        elif isinstance(item, str):
            if len(item.strip()) > 0:
                self.examples.append(item.strip())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize with truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


def create_bnb_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config."""
    if model_config.use_4bit:
        compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.use_nested_quant,
        )
    elif model_config.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def create_lora_config(model_config: ModelConfig) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias=model_config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(model_config: ModelConfig):
    """Load model and tokenizer with optional quantization."""
    logger.info(f"Loading model: {model_config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create quantization config
    bnb_config = create_bnb_config(model_config)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 if available
    )

    # Prepare model for k-bit training if using quantization
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # Apply LoRA
    lora_config = create_lora_config(model_config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def create_training_arguments(train_config: TrainConfig) -> TrainingArguments:
    """Create training arguments."""
    return TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        save_total_limit=train_config.save_total_limit,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        optim=train_config.optim,
        max_grad_norm=train_config.max_grad_norm,
        seed=train_config.seed,
        report_to=train_config.report_to if train_config.report_to != "none" else None,
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )


def train(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    eval_data_path: Optional[str] = None,
):
    """Main training function."""
    # Initialize wandb if enabled
    if train_config.report_to == "wandb":
        wandb.init(
            project="qwen3-dapt-lora",
            name=f"{model_config.model_name.split('/')[-1]}_dapt",
            config={
                "model": model_config.__dict__,
                "data": data_config.__dict__,
                "training": train_config.__dict__,
            },
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Create datasets
    logger.info("Loading training data...")
    train_dataset = JSONCorpusDataset(
        data_path=data_config.data_path,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
        text_field=data_config.text_field,
    )

    eval_dataset = None
    if eval_data_path:
        logger.info("Loading evaluation data...")
        eval_dataset = JSONCorpusDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=data_config.max_seq_length,
            text_field=data_config.text_field,
        )

    # Create data collator for language modeling (MLM=False for causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )

    # Create training arguments
    training_args = create_training_arguments(train_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {train_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(train_config.output_dir)

    # Save LoRA adapter separately
    lora_output_dir = os.path.join(train_config.output_dir, "lora_adapter")
    model.save_pretrained(lora_output_dir)
    logger.info(f"LoRA adapter saved to {lora_output_dir}")

    if train_config.report_to == "wandb":
        wandb.finish()

    return model, tokenizer


def inference_example(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Run inference with the trained model."""
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return response


def main():
    parser = argparse.ArgumentParser(description="SSL DAPT + LoRA Training for Qwen3")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B-Instruct"],
        help="Model to use for DAPT",
    )
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", default=False, help="Use 8-bit quantization")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (JSON/JSONL)")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Path to evaluation data")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--text_field", type=str, default="text", help="Field name containing text in JSON")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_dapt_lora", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")

    args = parser.parse_args()

    # Create configs
    model_config = ModelConfig(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    data_config = DataConfig(
        data_path=args.data_path,
        max_seq_length=args.max_seq_length,
        text_field=args.text_field,
    )

    train_config = TrainConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
    )

    # Run training
    model, tokenizer = train(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        eval_data_path=args.eval_data_path,
    )

    # Test inference
    logger.info("Testing inference...")
    response = inference_example(model, tokenizer, "Who are you?")
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
