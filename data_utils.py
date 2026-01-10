"""
Data Utilities for SSL DAPT Training

This module provides utilities for loading and preprocessing
JSON and JSONL corpus data for domain-adaptive pre-training.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import random

from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    source_file: str
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class JSONLIterator:
    """Memory-efficient iterator for large JSONL files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        text_field: str = "text",
        max_docs: Optional[int] = None,
    ):
        self.file_path = Path(file_path)
        self.text_field = text_field
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[DocumentChunk]:
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_docs and count >= self.max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = self._extract_text(data)
                    if text:
                        yield DocumentChunk(
                            text=text,
                            source_file=str(self.file_path),
                            doc_id=data.get("id"),
                            metadata={k: v for k, v in data.items() if k != self.text_field},
                        )
                        count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text from a JSON object."""
        # Try specified field first
        if self.text_field in data:
            return str(data[self.text_field]).strip()

        # Try common field names
        for field in ["text", "content", "body", "message", "document"]:
            if field in data:
                return str(data[field]).strip()

        # Fall back to concatenating string fields
        texts = [str(v) for v in data.values() if isinstance(v, str)]
        return " ".join(texts).strip() if texts else None


class StreamingJSONCorpusDataset(IterableDataset):
    """
    Streaming dataset for large JSON/JSONL corpus files.
    Useful when data doesn't fit in memory.
    """

    def __init__(
        self,
        data_paths: List[Union[str, Path]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        text_field: str = "text",
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        self.data_paths = [Path(p) for p in data_paths]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_field = text_field
        self.shuffle_files = shuffle_files
        self.seed = seed

        # Collect all files
        self.files = []
        for path in self.data_paths:
            if path.is_file():
                self.files.append(path)
            elif path.is_dir():
                self.files.extend(path.glob("**/*.json"))
                self.files.extend(path.glob("**/*.jsonl"))

        logger.info(f"Found {len(self.files)} data files")

    def __iter__(self):
        files = list(self.files)
        if self.shuffle_files:
            random.seed(self.seed)
            random.shuffle(files)

        for file_path in files:
            for chunk in JSONLIterator(file_path, self.text_field):
                encoding = self.tokenizer(
                    chunk.text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                    return_tensors=None,
                )
                yield {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                }


class PackedDataset(Dataset):
    """
    Dataset that packs multiple short documents into single sequences
    to maximize GPU utilization and training efficiency.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        pack_sequences: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        if pack_sequences:
            self._pack_texts(texts)
        else:
            self._tokenize_texts(texts)

        logger.info(f"Created {len(self.examples)} packed examples")

    def _tokenize_texts(self, texts: List[str]):
        """Tokenize texts without packing."""
        for text in texts:
            if not text.strip():
                continue
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )
            self.examples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            })

    def _pack_texts(self, texts: List[str]):
        """Pack multiple texts into sequences of max_seq_length."""
        # Tokenize all texts first
        all_tokens = []
        eos_token_id = self.tokenizer.eos_token_id

        for text in texts:
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(eos_token_id)  # Separator between documents

        # Pack into sequences
        for i in range(0, len(all_tokens), self.max_seq_length):
            chunk = all_tokens[i:i + self.max_seq_length]
            if len(chunk) > 10:  # Minimum sequence length
                self.examples.append({
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_json_corpus(
    data_path: Union[str, Path],
    text_field: str = "text",
    max_docs: Optional[int] = None,
) -> List[str]:
    """
    Load text from JSON/JSONL files.

    Supports:
    - Single JSON file with list of objects
    - Single JSON file with single object
    - JSONL file (one JSON object per line)
    - Directory containing JSON/JSONL files

    Args:
        data_path: Path to file or directory
        text_field: Field name containing text
        max_docs: Maximum number of documents to load

    Returns:
        List of text strings
    """
    data_path = Path(data_path)
    texts = []

    def process_item(item: Any) -> Optional[str]:
        if isinstance(item, str):
            return item.strip() if item.strip() else None
        if isinstance(item, dict):
            # Try specified field
            if text_field in item:
                return str(item[text_field]).strip()
            # Try common fields
            for field in ["text", "content", "body", "message"]:
                if field in item:
                    return str(item[field]).strip()
            # Concatenate string values
            string_vals = [str(v) for v in item.values() if isinstance(v, str)]
            return " ".join(string_vals).strip() if string_vals else None
        return None

    def load_file(file_path: Path):
        nonlocal texts
        if max_docs and len(texts) >= max_docs:
            return

        logger.info(f"Loading {file_path}")
        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_docs and len(texts) >= max_docs:
                        break
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            text = process_item(data)
                            if text:
                                texts.append(text)
                        except json.JSONDecodeError:
                            continue
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if max_docs and len(texts) >= max_docs:
                            break
                        text = process_item(item)
                        if text:
                            texts.append(text)
                else:
                    text = process_item(data)
                    if text:
                        texts.append(text)

    if data_path.is_file():
        load_file(data_path)
    elif data_path.is_dir():
        for file_path in sorted(data_path.glob("**/*.json")):
            load_file(file_path)
        for file_path in sorted(data_path.glob("**/*.jsonl")):
            load_file(file_path)
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    logger.info(f"Loaded {len(texts)} documents")
    return texts


def prepare_dapt_data(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    text_field: str = "text",
    pack_sequences: bool = True,
    train_test_split: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    Prepare data for DAPT training.

    Args:
        data_path: Path to JSON/JSONL data
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        text_field: Field containing text
        pack_sequences: Whether to pack short sequences
        train_test_split: Fraction for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load all texts
    texts = load_json_corpus(data_path, text_field)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(texts)

    split_idx = int(len(texts) * train_test_split)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    logger.info(f"Train: {len(train_texts)}, Eval: {len(eval_texts)}")

    # Create datasets
    train_dataset = PackedDataset(
        train_texts,
        tokenizer,
        max_seq_length,
        pack_sequences,
    )
    eval_dataset = PackedDataset(
        eval_texts,
        tokenizer,
        max_seq_length,
        pack_sequences,
    ) if eval_texts else None

    return train_dataset, eval_dataset


def create_sample_data(output_path: Union[str, Path], num_samples: int = 100):
    """Create sample JSONL data for testing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        {"text": f"This is sample document {i}. It contains some text for testing DAPT training."}
        for i in range(num_samples)
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Created {num_samples} samples at {output_path}")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("./data/sample_corpus.jsonl", num_samples=100)
