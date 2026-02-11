"""Custom collate function for instruction fine-tuning batches."""

import torch
from typing import List, Dict


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of tokenized examples.

    Pads sequences to batch max length, creates attention masks,
    and shifts targets by 1 for language modeling.

    Args:
        batch: List of dicts with input_ids and attention_mask

    Returns:
        Dict with batched and padded tensors
    """
    # TODO: Implement
    pass
