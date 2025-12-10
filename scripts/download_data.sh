#!/bin/bash
# scripts/download_data.sh

echo "🧬 STARTING GENOMIC DATA DOWNLOAD..."
mkdir -p data/raw

# 1. Download Drosophila Genome (dm6) from UCSC
# Size: ~40MB (Compressed)
echo "⬇️ Downloading dm6 Genome (DNA)..."
curl -o data/raw/dm6.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz

# 2. Download Fly Cell Atlas (10x Stringent - Whole Body)
# Size: ~2GB (Warning: Big File)
# Note: Using a placeholder link as FCA requires specific selection, 
# but this is where the .h5ad goes.
echo "⬇️ Downloading Fly Cell Atlas (Transcriptome)..."
echo "⚠️  NOTE: Please manually download the '10x Stringent' .h5ad file from https://flycellatlas.org/"
echo "    and place it in data/raw/fly_cell_atlas.h5ad"

echo "✅ Download script complete."