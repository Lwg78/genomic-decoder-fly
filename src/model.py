import torch
import torch.nn as nn
import math

class GenomicTransformer(nn.Module):
    """
    A Transformer model designed to read DNA sequences (K-Mer tokens)
    and predict regulatory activity (Gene Expression).
    """
    def __init__(self, vocab_size=69, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        
        # 1. Embedding Layer: Converts Integer IDs -> Vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Positional Encoding: DNA order matters!
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 3. Transformer Encoder: The "Brain"
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Prediction Head: Converts "Brain Output" -> "Gene Expression Value" (Regression)
        self.decoder = nn.Linear(embed_dim, 1) 

    def forward(self, x):
        # x shape: [Batch, Seq_Len] (e.g., 32, 126)
        
        # Embed and add position info
        x = self.embedding(x) * math.sqrt(128)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # output shape: [Batch, Seq_Len, Embed_Dim]
        x = self.transformer_encoder(x)
        
        # Pooling: We only care about the overall "meaning" of the sequence
        # We take the mean of all tokens
        x = x.mean(dim=1) 
        
        # Predict Expression Level
        output = self.decoder(x)
        return output

class PositionalEncoding(nn.Module):
    """Standard Transformer Positional Encoding (injects order info)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the embeddings
        return x + self.pe[:x.size(1), :]