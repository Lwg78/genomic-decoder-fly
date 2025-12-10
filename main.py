import os
import time
import torch
from torch.utils.data import DataLoader
from src.tokenizer import DNATokenizer
from src.dataloader import GenomicDataLoader
from src.dataset import GenomicDataset
from src.model import GenomicTransformer

# Configuration
K_MER_SIZE = 3
CONTEXT_WINDOW = 128   
BATCH_SIZE = 32        
TARGET_CHROMOSOME = "chr2L" 
MODEL_PATH = "genomic_model.pth"

def main():
    print("🧬 GENOMIC DECODER AI (FlyOS V1.0)")
    print("====================================")
    
    # 1. Initialize Components
    tokenizer = DNATokenizer(k=K_MER_SIZE)
    loader = GenomicDataLoader(data_dir="data/raw")

    # 2. Load Real Data (Lazy)
    print("\n[Phase 1] Data Ingestion")
    genome = loader.load_genome(gz_filename="dm6.fa.gz")
    
    dna_sequence = ""
    if genome and TARGET_CHROMOSOME in genome:
        print(f"    -> Accessing {TARGET_CHROMOSOME}...")
        dna_sequence = str(genome[TARGET_CHROMOSOME].seq).upper()
    else:
        print("⚠️ Using Synthetic DNA for demo.")
        dna_sequence = "ATGCGTACGTAGCTAGCTAGCTAGCTNNNATG" * 10000

    # 3. Build Dataset
    print("\n[Phase 2] Building Deep Learning Pipeline")
    dataset = GenomicDataset(dna_sequence, tokenizer, max_len=CONTEXT_WINDOW)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Initialize Model
    print("\n[Phase 3] Initializing Transformer Architecture")
    model = GenomicTransformer(vocab_size=len(tokenizer.vocab))
    print(f"    -> Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Training Simulation
    print("\n[Phase 4] Starting Training Loop (First 3 Batches)")
    
    device = torch.device("cpu") # Use "mps" if on M1/M2 Mac, "cpu" for Intel Mac
    model.to(device)
    model.train() 
    
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 3: break 
        
        batch = batch.to(device)
        prediction = model(batch)
        
        print(f"\n    🚀 Batch {i+1}:")
        print(f"       Input: {batch.shape} | Output: {prediction.shape}")
        print(f"       Sample Prediction: {prediction[0].item():.4f} (Expression Level)")
        
    print(f"\n✅ Speed Check: {BATCH_SIZE * 3} samples processed in {time.time() - start_time:.4f}s")
    
    # 6. Save State
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 Model Artifact saved to: {MODEL_PATH}")
    print("    -> Ready for Inference.")

if __name__ == "__main__":
    main()