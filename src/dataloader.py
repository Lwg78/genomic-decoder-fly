import os
import gzip
import shutil
import anndata
from Bio import SeqIO

class GenomicDataLoader:
    """
    Handles efficient loading of massive Genomic (DNA) and Transcriptomic (RNA) datasets.
    Optimized for low-RAM environments using Lazy Loading (Disk-Based Access).
    """
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        
    def _ensure_unzipped_genome(self, gz_filename="dm6.fa.gz"):
        """
        GZIP prevents random access. This helper extracts dm6.fa.gz -> dm6.fa
        so we can perform O(1) Lazy Loading.
        """
        gz_path = os.path.join(self.data_dir, gz_filename)
        fa_filename = gz_filename.replace(".gz", "") # dm6.fa
        fa_path = os.path.join(self.data_dir, fa_filename)

        # If .fa already exists, we are good.
        if os.path.exists(fa_path):
            return fa_filename

        # If not, but .gz exists, unzip it.
        if os.path.exists(gz_path):
            print(f"📦 First-time setup: Unzipping {gz_filename} to allow Lazy Loading...")
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(fa_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("    -> Unzip complete.")
                return fa_filename
            except Exception as e:
                print(f"❌ Error unzipping genome: {e}")
                return None
        
        return None

    def load_genome(self, gz_filename="dm6.fa.gz"):
        """
        Returns a Lazy-Loaded Genome Dictionary.
        Does NOT load DNA into RAM. Reads from disk on request.
        """
        # 1. Ensure we have the raw .fa file (not .gz)
        fa_filename = self._ensure_unzipped_genome(gz_filename)
        if not fa_filename:
            print(f"⚠️ Genome file not found in {self.data_dir}")
            return None

        filepath = os.path.join(self.data_dir, fa_filename)

        print(f"🧬 Lazy-Loading Genome from {fa_filename}...")
        try:
            # SeqIO.index creates a lookup table (Disk Pointer).
            # It uses < 1MB RAM but gives instant access to any chromosome.
            genome_index = SeqIO.index(filepath, "fasta")
            
            print(f"    -> Indexed {len(genome_index)} chromosomes (Zero RAM used).")
            return genome_index
            
        except Exception as e:
            print(f"❌ Error indexing genome: {e}")
            return None

    def load_transcriptome(self, filename="fly_cell_atlas.h5ad", backed_mode='r'):
        """
        Loads Single-Cell RNA-seq data (.h5ad) in Lazy Mode.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Transcriptome file not found at {filepath}")
            return None

        print(f"📊 Lazy-Loading Transcriptome from {filename}...")
        try:
            # backed='r' means we read from disk, not RAM.
            adata = anndata.read_h5ad(filepath, backed=backed_mode)
            print(f"    -> Connected to {adata.n_obs} cells x {adata.n_vars} genes")
            return adata
        except Exception as e:
            print(f"❌ Error loading transcriptome: {e}")
            return None