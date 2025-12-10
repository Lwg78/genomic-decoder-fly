# 🧬 Genomic Decoder (FlyOS)

> **A Transformer-based AI system designed to decode the regulatory logic of the Drosophila (Fruit Fly) genome.**

![Status](https://img.shields.io/badge/Status-Research%20Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Research Pipeline](https://github.com/Lwg78/genomic-decoder-fly/actions/workflows/run_research_pipeline.yml/badge.svg)

## ☁️ Automated Research Pipeline
This repository includes a CI/CD pipeline that proves the reproducibility of the experiment.
Every commit automatically:
1.  **Installs** the specific PyTorch/NumPy environment.
2.  **Downloads** the Drosophila genome from UCSC.
3.  **Trains** the Transformer model on a subset of data.
4.  **Publishes** the trained model weights as an artifact.

[View the Pipeline Logs](https://github.com/Lwg78/genomic-decoder-fly/actions)

## 🔬 Executive Summary
This project treats the cell not as a biological entity, but as a **Computational Operating System**. 

By applying Natural Language Processing (NLP) techniques—specifically **Transformers** and **K-Mer Tokenization**—to raw DNA sequences, this system aims to predict gene expression levels directly from the sequence, effectively "reading" the regulatory code of life.

**Core Concept:**
* **Hardware:** Chromatin Structure (Nucleosomes)
* **Software:** DNA Sequence (Promoters/Enhancers)
* **Compiler:** Neural Network (Genomic Decoder)

## 🛠 Architecture
The system handles massive genomic files using a **Lazy-Loading** strategy to run efficiently on consumer hardware (MacBook Air), despite the dataset size (23M+ base pairs).

```text
genomic-decoder-fly/
├── data/raw/             # 🛑 Stores dm6.fa (Lazy Loaded)
├── src/
│   ├── dataloader.py     # 🧬 Biopython-based O(1) Disk Access
│   ├── tokenizer.py      # 🔡 K-Mer Tokenization (Vocab Size: 69)
│   ├── dataset.py        # 📦 PyTorch Dataset (Sliding Window)
│   └── model.py          # 🧠 Transformer Encoder (1.2M Params)
├── scripts/
│   └── download_data.sh  # 📜 Auto-fetch UCSC Genomes
├── main.py               # 🚀 Training Loop Simulation
└── README.md             # 📄 Documentation
```

## 🚀 Key Technical Features

### 1. Lazy Loading (Big Data Engineering)
* **Problem:** The Drosophila genome and Single-Cell Atlas are gigabytes in size. Loading them into RAM crashes standard laptops.
* **Solution:** Implemented `GenomicDataLoader` using **Biopython Indexing**. This allows O(1) random access to any chromosome directly from the disk, keeping RAM usage near zero.

### 2. K-Mer Tokenization (NLP for Biology)
* Instead of One-Hot Encoding (A,C,G,T), we use **K-Mers** (e.g., `ATG`, `TGC`).
* This captures local context, similar to how Words are more meaningful than Letters in English.
* **Vocabulary:** 64 combinations + Special Tokens (`[CLS]`, `[PAD]`).

### 3. Transformer Backbone
* We utilize a custom `GenomicTransformer` with Positional Encoding.
* **Parameters:** ~1.2 Million.
* **Task:** Regression (Predicting Transcriptome abundance from DNA).

## ⚙️ How to Run

1.  **Install Dependencies:**
    ```bash
    pip install torch biopython anndata numpy<2.0.0
    ```

2.  **Download Data:**
    ```bash
    bash scripts/download_data.sh
    ```

3.  **Run Pipeline:**
    ```bash
    python main.py
    ```

## 🔮 Future Roadmap
* **Phase 2:** Integrate `fly_cell_atlas.h5ad` labels to train the model on real Gene Expression data.
* **Phase 3:** Implement "In-Silico Mutagenesis" to predict how mutations affect gene regulation.

---
**Author:** Lim Wen Gio  
*Research Prototype for AI Portfolio.*
