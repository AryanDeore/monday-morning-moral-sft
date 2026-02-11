"""InstructionDataset: Loads curated dataset and tokenizes at initialization."""

from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from typing import Optional


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning.
    
    Loads curated TinyStories-Instruct dataset, formats with template,
    and tokenizes each example at initialization.
    """

    def __init__(self, split: str = "train", max_length: int = 512):
        """Initialize dataset.

        Args:
            split: "train" or "test"
            max_length: Maximum token sequence length (truncate if longer)
        """
        # TODO: Implement
        pass

    def __len__(self) -> int:
        # TODO: Implement
        return 0

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized example.

        Returns:
            Dict with keys: input_ids, attention_mask
        """
        # TODO: Implement
        pass
