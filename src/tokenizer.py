import numpy as np
import itertools

class DNATokenizer:
    """
    Converts raw DNA strings into numerical tokens for the AI.
    Uses K-Mer Tokenization (splitting DNA into overlapping words).
    """
    def __init__(self, k=3):
        self.k = k
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        """Generates all possible combinations (e.g., AAA, AAC...)"""
        bases = ['A', 'C', 'G', 'T', 'N'] # N = Unknown
        kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        # Map kmer to integer ID
        return {kmer: i for i, kmer in enumerate(kmers)}

    def encode(self, sequence):
        """
        Input: "ATCGG..."
        Output: [0, 4, 12, ...] (Vector of IDs)
        """
        sequence = sequence.upper()
        tokens = []
        # Sliding window strategy
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            if kmer in self.vocab:
                tokens.append(self.vocab[kmer])
            else:
                tokens.append(0) # Unknown token
        return np.array(tokens)

    def decode(self, token_ids):
        """Reverses IDs back to DNA (for generation tasks)"""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return [reverse_vocab.get(tid, "NNN") for tid in token_ids]