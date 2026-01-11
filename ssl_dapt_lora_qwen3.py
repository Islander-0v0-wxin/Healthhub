"""
SSL DAPT + LoRA Training Script for Qwen3 Models

Domain-Adaptive Pre-Training (DAPT) with LoRA on Qwen3-4B or Qwen3-8B base models.

DAPT Philosophy:
- Continue pre-training on domain-specific corpus to inject new knowledge
- Use BASE models (not instruct) for better knowledge absorption
- Lower learning rates to prevent catastrophic forgetting
- Longer training with gradual warmup for stable learning
- Full precision training recommended for best quality (quantization optional)

Key DAPT Best Practices:
1. Use base models, not instruction-tuned versions
2. Lower learning rate (1e-5 to 5e-5) compared to fine-tuning
3. Longer warmup (5-10% of total steps)
4. Train for multiple epochs on domain corpus
5. Use sequence packing for efficiency
6. Higher LoRA rank for more capacity to learn new knowledge
"""

import os
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
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

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model and LoRA settings optimized for DAPT."""
    # Use BASE model for DAPT (not instruct versions)
    model_name: str = "Qwen/Qwen3-4B"

    # Quantization - OFF by default for best training quality
    # Enable only if GPU memory is limited
    use_quantization: bool = False
    quantization_bits: int = 4  # 4 or 8
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # LoRA configuration - Higher rank for DAPT to learn more knowledge
    lora_r: int = 128  # Higher rank for more capacity
    lora_alpha: int = 256  # 2x rank is common
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"

    # Precision
    torch_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str = "./data"
    max_seq_length: int = 2048
    text_field: str = "text"
    pack_sequences: bool = True  # Pack short docs for efficiency
    min_sequence_length: int = 64  # Minimum tokens per sequence


@dataclass
class DAPTTrainConfig:
    """Training configuration optimized for Domain-Adaptive Pre-Training."""
    output_dir: str = "./outputs/qwen3_dapt_lora"

    # Hugging Face Hub settings (for saving to remote)
    push_to_hub: bool = False
    hub_model_id: str = None  # e.g., "username/model-name"
    hub_token: str = None  # HF token, or set HF_TOKEN env var
    hub_private: bool = True  # Make repo private

    # DAPT typically needs more epochs for knowledge injection
    num_train_epochs: int = 3

    # Batch settings
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Larger effective batch

    # DAPT uses LOWER learning rate to prevent catastrophic forgetting
    learning_rate: float = 5e-5  # Lower than fine-tuning (typically 1e-5 to 5e-5)
    min_learning_rate: float = 1e-6  # Minimum LR for cosine decay

    # Longer warmup for stable DAPT
    warmup_ratio: float = 0.1  # 10% warmup

    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Memory optimization
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"  # Use standard AdamW for non-quantized

    seed: int = 42
    report_to: str = "none"


# =============================================================================
# Dataset Classes
# =============================================================================

class DAPTCorpusDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-Training with sequence packing.

    Packing concatenates multiple short documents into single sequences
    to maximize GPU utilization and training efficiency.
    """

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
        self.pack_sequences = pack_sequences
        self.min_sequence_length = min_sequence_length
        self.examples = []

        # Load raw texts
        texts = self._load_texts(data_path)
        logger.info(f"Loaded {len(texts)} documents from {data_path}")

        # Process into training examples
        if pack_sequences:
            self._pack_texts(texts)
        else:
            self._tokenize_texts(texts)

        logger.info(f"Created {len(self.examples)} training examples")

    def _load_texts(self, data_path: str) -> List[str]:
        """Load texts from JSON/JSONL files."""
        data_path = Path(data_path)
        texts = []

        def process_item(item: Any) -> Optional[str]:
            if isinstance(item, str):
                return item.strip() if item.strip() else None
            if isinstance(item, dict):
                # Try specified field first
                if self.text_field in item:
                    return str(item[self.text_field]).strip()
                # Try common fields
                for field in ["text", "content", "body", "document", "passage"]:
                    if field in item:
                        return str(item[field]).strip()
                # Concatenate all string values
                strings = [str(v) for v in item.values() if isinstance(v, str)]
                return " ".join(strings).strip() if strings else None
            return None

        def load_file(file_path: Path):
            logger.info(f"Loading {file_path}")
            try:
                if file_path.suffix == ".jsonl":
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    text = process_item(data)
                                    if text and len(text) > 10:
                                        texts.append(text)
                                except json.JSONDecodeError:
                                    continue
                else:  # .json
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                text = process_item(item)
                                if text and len(text) > 10:
                                    texts.append(text)
                        else:
                            text = process_item(data)
                            if text and len(text) > 10:
                                texts.append(text)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

        if data_path.is_file():
            load_file(data_path)
        elif data_path.is_dir():
            for pattern in ["**/*.json", "**/*.jsonl"]:
                for file_path in sorted(data_path.glob(pattern)):
                    load_file(file_path)
        else:
            raise ValueError(f"Data path does not exist: {data_path}")

        return texts

    def _tokenize_texts(self, texts: List[str]):
        """Tokenize texts without packing."""
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )
            if len(encoding["input_ids"]) >= self.min_sequence_length:
                self.examples.append({
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                })

    def _pack_texts(self, texts: List[str]):
        """
        Pack multiple documents into sequences of max_seq_length.

        This improves training efficiency by reducing padding and
        ensuring each batch processes maximum tokens.
        """
        # Shuffle texts for better mixing
        random.shuffle(texts)

        # Tokenize all texts
        all_tokens = []
        eos_token_id = self.tokenizer.eos_token_id

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(eos_token_id)  # Document separator

        # Pack into fixed-length sequences
        for i in range(0, len(all_tokens), self.max_seq_length):
            chunk = all_tokens[i:i + self.max_seq_length]
            if len(chunk) >= self.min_sequence_length:
                self.examples.append({
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Model Loading
# =============================================================================

def create_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config if enabled."""
    if not model_config.use_quantization:
        return None

    if model_config.quantization_bits == 4:
        compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.use_nested_quant,
        )
    elif model_config.quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def create_lora_config(model_config: ModelConfig) -> LoraConfig:
    """Create LoRA configuration optimized for DAPT."""
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias=model_config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(model_config: ModelConfig, train_config: DAPTTrainConfig):
    """
    Load model and tokenizer for DAPT.

    For best DAPT results:
    - Use base models (not instruct versions)
    - Full precision training when possible
    - Enable gradient checkpointing for memory efficiency
    """
    logger.info(f"Loading model: {model_config.model_name}")

    # Check if using base model (recommended for DAPT)
    if "Instruct" in model_config.model_name or "Chat" in model_config.model_name:
        logger.warning(
            "Using instruction-tuned model for DAPT. "
            "Consider using base model for better knowledge absorption."
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config (optional)
    bnb_config = create_quantization_config(model_config)

    # Model loading kwargs
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        logger.info(f"Using {model_config.quantization_bits}-bit quantization")
    else:
        # Full precision for best DAPT quality
        torch_dtype = getattr(torch, model_config.torch_dtype)
        model_kwargs["torch_dtype"] = torch_dtype
        logger.info(f"Using full precision ({model_config.torch_dtype})")

    # Try Flash Attention 2
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except Exception:
        logger.info("Flash Attention 2 not available, using default attention")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        **model_kwargs
    )

    # Prepare for quantized training if needed
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_config.gradient_checkpointing,
        )
    elif train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = create_lora_config(model_config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def create_training_arguments(
    train_config: DAPTTrainConfig,
    use_quantization: bool = False
) -> TrainingArguments:
    """Create training arguments optimized for DAPT."""

    # Use paged optimizer for quantized training
    optim = "paged_adamw_32bit" if use_quantization else train_config.optim

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
        optim=optim,
        max_grad_norm=train_config.max_grad_norm,
        seed=train_config.seed,
        report_to=train_config.report_to if train_config.report_to != "none" else None,
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_safetensors=True,
    )


def train_dapt(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: DAPTTrainConfig,
    eval_data_path: Optional[str] = None,
):
    """
    Main DAPT training function.

    Domain-Adaptive Pre-Training continues pre-training the model
    on domain-specific corpus to inject new knowledge while
    preserving existing capabilities.
    """
    # Initialize wandb if enabled
    if train_config.report_to == "wandb":
        try:
            import wandb
            wandb.init(
                project="qwen3-dapt-lora",
                name=f"{model_config.model_name.split('/')[-1]}_dapt",
                config={
                    "model": model_config.__dict__,
                    "data": data_config.__dict__,
                    "training": train_config.__dict__,
                },
            )
        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            train_config.report_to = "none"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config, train_config)

    # Create training dataset
    logger.info("Loading training data...")
    train_dataset = DAPTCorpusDataset(
        data_path=data_config.data_path,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
        text_field=data_config.text_field,
        pack_sequences=data_config.pack_sequences,
        min_sequence_length=data_config.min_sequence_length,
    )

    # Create eval dataset if provided
    eval_dataset = None
    if eval_data_path:
        logger.info("Loading evaluation data...")
        eval_dataset = DAPTCorpusDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=data_config.max_seq_length,
            text_field=data_config.text_field,
            pack_sequences=data_config.pack_sequences,
            min_sequence_length=data_config.min_sequence_length,
        )

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = create_training_arguments(
        train_config,
        use_quantization=model_config.use_quantization
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Log training info
    logger.info("=" * 60)
    logger.info("DAPT Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Quantization: {'Enabled (' + str(model_config.quantization_bits) + '-bit)' if model_config.use_quantization else 'Disabled (full precision)'}")
    logger.info(f"LoRA rank: {model_config.lora_r}")
    logger.info(f"Learning rate: {train_config.learning_rate}")
    logger.info(f"Warmup ratio: {train_config.warmup_ratio}")
    logger.info(f"Epochs: {train_config.num_train_epochs}")
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Sequence packing: {data_config.pack_sequences}")
    logger.info("=" * 60)

    # Train
    logger.info("Starting DAPT training...")
    trainer.train()

    # Save final model locally
    logger.info(f"Saving model to {train_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(train_config.output_dir)

    # Save LoRA adapter separately
    lora_output_dir = os.path.join(train_config.output_dir, "lora_adapter")
    model.save_pretrained(lora_output_dir)
    logger.info(f"LoRA adapter saved to {lora_output_dir}")

    # Push to Hugging Face Hub if configured
    if train_config.push_to_hub and train_config.hub_model_id:
        logger.info(f"Pushing model to Hugging Face Hub: {train_config.hub_model_id}")
        try:
            # Get token from config or environment
            token = train_config.hub_token or os.environ.get("HF_TOKEN")

            # Push LoRA adapter to Hub
            model.push_to_hub(
                train_config.hub_model_id,
                token=token,
                private=train_config.hub_private,
                commit_message="DAPT LoRA adapter for domain knowledge injection",
            )

            # Push tokenizer
            tokenizer.push_to_hub(
                train_config.hub_model_id,
                token=token,
                private=train_config.hub_private,
            )

            logger.info(f"Successfully pushed to: https://huggingface.co/{train_config.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
            logger.info("Model saved locally. You can manually push later.")

    # Finish wandb
    if train_config.report_to == "wandb":
        try:
            import wandb
            wandb.finish()
        except:
            pass

    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    """Generate text with the DAPT-trained model."""
    # For base models, just use the text directly
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSL DAPT + LoRA Training for Qwen3 - Domain-Adaptive Pre-Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic DAPT training (recommended - full precision)
  python ssl_dapt_lora_qwen3.py --data_path ./data/corpus.jsonl

  # Save to custom directory (e.g., mounted drive)
  python ssl_dapt_lora_qwen3.py --data_path ./data --output_dir /mnt/storage/models/qwen3_dapt

  # Push to Hugging Face Hub after training
  python ssl_dapt_lora_qwen3.py --data_path ./data --push_to_hub --hub_model_id username/my-dapt-model

  # DAPT with 4-bit quantization (for limited GPU memory)
  python ssl_dapt_lora_qwen3.py --data_path ./data --use_quantization --quantization_bits 4

  # DAPT on Qwen3-8B with custom settings
  python ssl_dapt_lora_qwen3.py \\
      --model_name Qwen/Qwen3-8B \\
      --data_path ./medical_corpus \\
      --lora_r 256 \\
      --learning_rate 2e-5 \\
      --num_epochs 5
        """
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model for DAPT (use base models, not instruct versions)",
    )

    # Quantization (optional)
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=False,
        help="Enable quantization (default: off for best quality)",
    )
    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits if enabled (default: 4)",
    )

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank (higher = more capacity)")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (JSON/JSONL)")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Path to evaluation data")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--text_field", type=str, default="text", help="Field name containing text")
    parser.add_argument("--no_packing", action="store_true", help="Disable sequence packing")

    # Training arguments (DAPT-optimized defaults)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_dapt_lora", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (lower for DAPT)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (longer for DAPT)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")

    # Hugging Face Hub arguments (for remote saving)
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub after training")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub model ID (e.g., 'username/model-name')")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--hub_private", action="store_true", default=True, help="Make Hub repo private (default: True)")

    args = parser.parse_args()

    # Create configs
    model_config = ModelConfig(
        model_name=args.model_name,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    data_config = DataConfig(
        data_path=args.data_path,
        max_seq_length=args.max_seq_length,
        text_field=args.text_field,
        pack_sequences=not args.no_packing,
    )

    train_config = DAPTTrainConfig(
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        hub_private=args.hub_private,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
    )

    # Run DAPT training
    model, tokenizer = train_dapt(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        eval_data_path=args.eval_data_path,
    )

    # Test generation
    logger.info("Testing generation with trained model...")
    test_prompt = "The human body"
    response = generate_text(model, tokenizer, test_prompt, max_new_tokens=100)
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Generated: {response}")


if __name__ == "__main__":
    main()
