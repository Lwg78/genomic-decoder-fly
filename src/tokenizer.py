import itertools
import numpy as np

class DNATokenizer:
    """
    Research-Grade DNA Tokenizer (K-Mer Strategy).
    Compatible with Transformer architectures (BERT/GPT).
    
    Vocabulary Structure:
    - 0: [PAD]  (Padding)
    - 1: [UNK]  (Unknown / 'N' bases)
    - 2: [CLS]  (Start of Sequence)
    - 3: [SEP]  (End of Sequence)
    - 4: [MASK] (For Pre-training)
    - 5...N: K-Mers (e.g., 'AAA', 'AAC'...)
    """
    def __init__(self, k=3, stride=1):
        self.k = k
        self.stride = stride
        
        # 1. Define Special Tokens
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4
        }
        
        # 2. Build Vocabulary (4^k combinations)
        self.vocab = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        bases = ['A', 'C', 'G', 'T']
        kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        
        # Start IDs after special tokens
        start_id = len(self.special_tokens)
        for i, kmer in enumerate(kmers):
            self.vocab[kmer] = start_id + i
            self.id_to_token[start_id + i] = kmer
            
        print(f"✅ Tokenizer Ready: K={k}, Stride={stride}, Vocab Size={len(self.vocab)}")

    def encode(self, sequence, add_special_tokens=True):
        """
        Converts DNA string -> List of Integer IDs.
        Handles 'N' or invalid chars by mapping to [UNK].
        """
        if not sequence:
            return np.array([])

        sequence = sequence.upper()
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab["[CLS]"])
            
        # Sliding Window Logic
        # Range: stops at len - k + 1 to ensure last window is full size
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i+self.k]
            
            # Check if valid (only ACGT)
            # If it contains 'N', map to [UNK]
            if all(b in "ACGT" for b in kmer):
                tokens.append(self.vocab[kmer])
            else:
                tokens.append(self.vocab["[UNK]"])
                
        if add_special_tokens:
            tokens.append(self.vocab["[SEP]"])
            
        return np.array(tokens)

    def decode(self, token_ids):
        """Reverses IDs -> List of K-Mers"""
        return [self.id_to_token.get(tid, "[UNK]") for tid in token_ids]

    def save_vocab(self, filepath):
        """Saves vocabulary for reproducibility"""
        with open(filepath, 'w') as f:
            for token, id in self.vocab.items():
                f.write(f"{token}\t{id}\n")