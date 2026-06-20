import os
import torch
from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder


class HDFA_ProjectHarvester:
    def __init__(self, target_directory, file_extensions=None):
        """
        Initializes a workspace folder file crawler to pump raw codebase blocks 
        directly into the hyperdimensional character sequence encoder.
        """
        self.target_dir = target_directory
        self.extensions = file_extensions or ['.py', '.js', '.jsx', '.html', '.css']
        self.processed_files_count = 0

    def harvest_and_stream_workspace(self, vector_engine, sliding_encoder):
        """
        Crawls local directories, safely skips heavy environment junk, reads active code, 
        and pipes character wave blocks straight into memory cache.
        """
        print(f"[HARVESTER] Initiating workspace filesystem scan on: {self.target_dir}")
        global_sequence_stream = []

        for root, dirs, files in os.walk(self.target_dir):
            # Bypass heavy runtime node caches to optimize laptop CPU cache lines
            if any(folder in root for folder in ['node_modules', '.git', '__pycache__', '.pytest_cache', 'dist']):
                continue

            for file in files:
                if any(file.endswith(ext) for ext in self.extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read().strip()
                            
                            if len(file_content) > 10:
                                # Process raw source file text through your Day 8 sliding window script
                                file_waves = sliding_encoder.encode_file_stream(file_content)
                                global_sequence_stream.append(file_waves)
                                self.processed_files_count += 1
                                print(f"[HARVESTER] Synthesized character wave tensor for: {file} | Shape: {file_waves.shape}")
                    except Exception:
                        # Silently skip locked, binary, or non-utf-8 configurations
                        continue

        print(f"\n[SUCCESS] Workspace Ingestion Complete! Crawled {self.processed_files_count} clean project script assets.")
        
        if len(global_sequence_stream) > 0:
            # Concat everything into one unified, master spatiotemporal data block
            return torch.cat(global_sequence_stream, dim=0)
        else:
            return torch.zeros(0, vector_engine.dimension)

# --- DIRECTORY INTEGRATION VALIDATION TEST ---
if __name__ == "__main__":
    print("Testing Phase 2 System Ingestion Integration...")
    
    # 1. Initialize foundational layers
    engine = HDC_VectorEngine()
    encoder = HDFA_SlidingEncoder(engine, window_size=3)
    
    # 2. Target your local hdfa-core directory to let the engine digest its own codebase!
    current_workspace = os.path.dirname(os.path.abspath(__file__))
    harvester = HDFA_ProjectHarvester(current_workspace, file_extensions=['.py'])
    
    # 3. Stream workspace arrays straight into a memory pool
    master_tensor_pool = harvester.harvest_and_stream_workspace(engine, encoder)
    print(f"Total compiled project hypervector wave arrays loaded in memory: {master_tensor_pool.shape}")
