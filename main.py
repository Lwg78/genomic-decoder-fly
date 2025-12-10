import os
import time
import numpy as np
from src.tokenizer import DNATokenizer

# Configuration
K_MER_SIZE = 3
EXAMPLE_GENE = "ATGCGTACGTAGCTAGCTAGCTAGCTNNNATG" # Fake gene for testing

def main():
    print("🧬 GENOMIC DECODER AI (FlyOS V1.0)")
    print("====================================")
    
    # 1. Initialize the Tokenizer
    print(f"[1] Initializing DNA K-Mer Tokenizer (K={K_MER_SIZE})...")
    tokenizer = DNATokenizer(k=K_MER_SIZE)
    print(f"    -> Vocabulary Size: {len(tokenizer.vocab)} unique DNA words")

    # 2. Simulate Loading Data
    # (In the future, this calls dataloader.py to read dm6.fa)
    print(f"[2] Loading Genomic Sequence (Simulation)...")
    print(f"    -> Input Sequence: {EXAMPLE_GENE[:15]}...")
    
    # 3. Encoding (The "Reading" Process)
    print("[3] AI is reading the DNA...")
    time.sleep(1) # Dramatic pause for demo
    encoded_vector = tokenizer.encode(EXAMPLE_GENE)
    print(f"    -> Encoded Tensor: {encoded_vector[:10]}... (Length: {len(encoded_vector)})")

    # 4. Decoding Logic (The Research Goal)
    # Your paper says: "Predict cellular behavior"
    # Here we simulate a "Gene Expression Prediction"
    print("[4] Decoding Regulatory Logic...")
    time.sleep(1)
    
    # Mock Prediction: GC-Rich regions often indicate high expression
    gc_content = (EXAMPLE_GENE.count('G') + EXAMPLE_GENE.count('C')) / len(EXAMPLE_GENE)
    predicted_expression = gc_content * 100 + np.random.normal(0, 5)
    
    print(f"✅ DECODING COMPLETE")
    print(f"    -> Predicted Gene Expression Level: {predicted_expression:.2f} TPM")
    print(f"    -> Regulatory Context: {'Active Promoter' if gc_content > 0.5 else 'Silenced Region'}")

if __name__ == "__main__":
    main()