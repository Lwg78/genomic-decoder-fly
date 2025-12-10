import torch
from torch.utils.data import Dataset
import numpy as np

class GenomicDataset(Dataset):
    """
    PyTorch Dataset that slices a massive DNA sequence into training windows.
    Strategy: Sliding Window (Non-overlapping for simplicity).
    """
    def __init__(self, dna_sequence, tokenizer, max_len=512):
        """
        Args:
            dna_sequence (str): The massive chromosome string (e.g. "ATCG...")
            tokenizer (DNATokenizer): Your tokenizer object
            max_len (int): How long each training chunk should be (Context Window)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dna_sequence = dna_sequence
        
        # Calculate how many full samples we can extract
        # Total Length // Window Size
        self.num_samples = len(dna_sequence) // max_len
        print(f"📦 Dataset Created: {self.num_samples:,} training samples (Length: {max_len})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Fetches the i-th window of DNA, tokenizes it, and returns a Tensor.
        """
        # 1. Calculate start and end indices
        start_idx = idx * self.max_len
        end_idx = start_idx + self.max_len
        
        # 2. Slice the raw string (Very fast)
        raw_snippet = self.dna_sequence[start_idx:end_idx]
        
        # 3. Tokenize
        # We don't add special tokens inside the window to keep size fixed
        token_ids = self.tokenizer.encode(raw_snippet, add_special_tokens=False)
        
        # 4. Convert to PyTorch Tensor
        # LongTensor is required for Embedding layers
        return torch.tensor(token_ids, dtype=torch.long)