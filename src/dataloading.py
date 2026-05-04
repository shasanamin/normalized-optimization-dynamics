"""Data loading helpers for release experiments.

The image experiments keep their compact in-script loaders for historical
compatibility. This module adds the cached causal-LM data flow used by the
Transformer / LayerNorm validation runs.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class CausalLMDataset(Dataset):
    """Fixed-length token chunks for next-token language modeling."""

    def __init__(self, hf_split_dataset):
        self.ds = hf_split_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.ds[index]
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return ids[:-1], ids[1:]


def _require_hf_dependencies():
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformer LM data loading requires `datasets` and `transformers`. "
            "Install them with `pip install -r supplementary_material/requirements.txt`."
        ) from exc
    return DatasetDict, load_dataset, load_from_disk, AutoTokenizer


def _tokenize_and_group(raw, tokenizer, block_size: int, num_proc: int, desc: str):
    DatasetDict, _, _, _ = _require_hf_dependencies()

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw["train"].column_names,
        num_proc=num_proc,
        desc=f"Tokenizing {desc}",
    )

    eos_id = tokenizer.eos_token_id
    chunk_len = block_size + 1

    def group_texts(examples):
        merged = []
        for ids in examples["input_ids"]:
            if ids:
                merged.extend(ids)
                merged.append(eos_id)

        total_length = (len(merged) // chunk_len) * chunk_len
        merged = merged[:total_length]
        return {
            "input_ids": [
                merged[i:i + chunk_len]
                for i in range(0, total_length, chunk_len)
            ]
        }

    processed = DatasetDict()
    for split in tokenized.keys():
        processed[split] = tokenized[split].map(
            group_texts,
            batched=True,
            batch_size=1000,
            remove_columns=tokenized[split].column_names,
            num_proc=num_proc,
            desc=f"Grouping tokens for {split}",
        )
    return processed


def prepare_wikitext_gpt2_pipeline(
    save_dir: str,
    dataset_config: str = "wikitext-103-raw-v1",
    tokenizer_name: str = "gpt2",
    block_size: int = 256,
    hf_cache_dir: Optional[str] = None,
    num_proc: int = 1,
) -> dict:
    """Tokenize WikiText and save a reusable causal-LM dataset cache."""
    _, load_dataset, _, AutoTokenizer = _require_hf_dependencies()

    save_path = Path(save_dir)
    tokenizer_dir = save_path / "tokenizer"
    dataset_dir = save_path / "dataset"
    cache_dir = Path(hf_cache_dir) if hf_cache_dir else save_path / "hf_cache"
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(tokenizer_dir)

    raw = load_dataset(
        "Salesforce/wikitext",
        dataset_config,
        cache_dir=cache_dir,
    )
    processed = _tokenize_and_group(raw, tokenizer, block_size, num_proc, "WikiText")
    processed.save_to_disk(dataset_dir)

    return {
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": dataset_config,
        "tokenizer_name": tokenizer_name,
        "block_size": block_size,
        "tokenizer_dir": str(tokenizer_dir),
        "dataset_dir": str(dataset_dir),
    }


def prepare_openweb_gpt2_pipeline(
    save_dir: str,
    tokenizer_name: str = "gpt2",
    block_size: int = 256,
    hf_cache_dir: Optional[str] = None,
    num_proc: int = 1,
    val_size: float = 0.001,
    split_seed: int = 42,
) -> dict:
    """Tokenize OpenWebText and save a reusable causal-LM dataset cache."""
    DatasetDict, load_dataset, _, AutoTokenizer = _require_hf_dependencies()

    save_path = Path(save_dir)
    tokenizer_dir = save_path / "tokenizer"
    dataset_dir = save_path / "dataset"
    cache_dir = Path(hf_cache_dir) if hf_cache_dir else save_path / "hf_cache"
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(tokenizer_dir)

    raw_train = load_dataset("openwebtext", split="train", cache_dir=cache_dir)
    split = raw_train.train_test_split(test_size=val_size, seed=split_seed)
    raw = DatasetDict({"train": split["train"], "validation": split["test"]})
    processed = _tokenize_and_group(raw, tokenizer, block_size, num_proc, "OpenWebText")
    processed.save_to_disk(dataset_dir)

    return {
        "dataset_name": "openwebtext",
        "tokenizer_name": tokenizer_name,
        "block_size": block_size,
        "tokenizer_dir": str(tokenizer_dir),
        "dataset_dir": str(dataset_dir),
        "val_size": val_size,
        "split_seed": split_seed,
    }


def is_prepared(save_dir: str) -> bool:
    save_path = Path(save_dir)
    tokenizer_dir = save_path / "tokenizer"
    dataset_dir = save_path / "dataset"
    tokenizer_ok = tokenizer_dir.is_dir() and (
        tokenizer_dir / "tokenizer_config.json"
    ).exists()
    dataset_ok = dataset_dir.is_dir() and (
        (dataset_dir / "dataset_dict.json").exists()
        or (dataset_dir / "dataset_info.json").exists()
    )
    return tokenizer_ok and dataset_ok


def _load_split(
    save_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
):
    _, _, load_from_disk, AutoTokenizer = _require_hf_dependencies()
    save_path = Path(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_path / "tokenizer")
    ds_all = load_from_disk(save_path / "dataset")
    ds_split = ds_all[split]
    ds_split.set_format(type="python", columns=["input_ids"])
    dataset = CausalLMDataset(ds_split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return tokenizer, dataset, loader


def load_wikitext(
    save_dir: str,
    batch_size: int,
    block_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prepare_if_missing: bool = True,
):
    """Return tokenizer, train/validation loaders, and datasets for WikiText."""
    if not is_prepared(save_dir):
        if not prepare_if_missing:
            raise FileNotFoundError(f"Prepared WikiText cache not found at {save_dir}")
        prepare_wikitext_gpt2_pipeline(save_dir=save_dir, block_size=block_size)

    tokenizer, train_dataset, train_loader = _load_split(
        save_dir, "train", batch_size, num_workers, pin_memory, persistent_workers
    )
    _, val_dataset, val_loader = _load_split(
        save_dir, "validation", batch_size, num_workers, pin_memory, persistent_workers
    )
    return tokenizer, train_loader, val_loader, train_dataset, val_dataset


def load_openweb(
    save_dir: str,
    batch_size: int,
    block_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prepare_if_missing: bool = True,
):
    """Return tokenizer, train/validation loaders, and datasets for OpenWebText."""
    if not is_prepared(save_dir):
        if not prepare_if_missing:
            raise FileNotFoundError(f"Prepared OpenWebText cache not found at {save_dir}")
        prepare_openweb_gpt2_pipeline(save_dir=save_dir, block_size=block_size)

    tokenizer, train_dataset, train_loader = _load_split(
        save_dir, "train", batch_size, num_workers, pin_memory, persistent_workers
    )
    _, val_dataset, val_loader = _load_split(
        save_dir, "validation", batch_size, num_workers, pin_memory, persistent_workers
    )
    return tokenizer, train_loader, val_loader, train_dataset, val_dataset
