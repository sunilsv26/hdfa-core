import torch
import time
import os
import sys

from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder
from .repo_harvester import HDFA_ProjectHarvester
from .fluid_grid import HDFA_FluidGrid
from .save_state import HDFA_MemorySaver

class HDFA_RepoTrainer:
    def __init__(self):
        print("[PIPELINE] Initializing High-Resolution Training Architecture...")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        self.grid = HDFA_FluidGrid()
        self.saver = HDFA_MemorySaver(self.engine)

    def execute_one_shot_training(self, workspace_path):
        print(f"[HARVESTER] Initiating high-resolution line extraction on: {workspace_path}")
        
        # We will harvest code files and break them down into discrete lines
        extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json']
        learned_lines_cache = {}
        processed_files_count = 0

        for root, dirs, files in os.walk(workspace_path):
            ignored = ['node_modules', '.git', '__pycache__', '.pytest_cache', 'dist', 'build', '.next', 'out']
            dirs[:] = [d for d in dirs if d not in ignored]

            for file in files:
                if file in ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'package.json']:
                    continue
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                clean_line = line.strip()
                                # Only memorize meaningful lines of code
                                if len(clean_line) > 10 and not clean_line.startswith(('import', 'from', '*', '//')):
                                    # Project line into a highly unique hyperdimensional vector shape
                                    line_waves = self.encoder.encode_file_stream(clean_line)
                                    if line_waves is not None and line_waves.shape[0] > 0:
                                        line_vector = torch.sign(torch.sum(line_waves, dim=0))
                                        line_vector[line_vector == 0] = -1.0
                                        
                                        # Cache the raw line text tied directly to its mathematical identifier
                                        learned_lines_cache[clean_line] = line_vector
                        processed_files_count += 1
                        print(f"[LEARNED] Extracted high-res lines from: {file}")
                    except Exception:
                        continue

        if len(learned_lines_cache) == 0:
            print("[ERROR] No unique code sequences discovered.")
            return

        print(f"\n[SUCCESS] Extracted {len(learned_lines_cache)} high-resolution code structures!")

        # Inject the entire learned dictionary directly into the engine's active codebook memory
        self.engine.codebook = {**self.engine.codebook, **learned_lines_cache}
        
        destination_brain_path = os.path.normpath(os.path.join(workspace_path, "codebase_brain.pt"))
        print(f"\n[STEP 2] Committing high-res knowledge base to drive...")
        self.saver.save_brain_snapshot(destination_brain_path)
        print(f"[SUCCESS] High-resolution brain snapshot sealed at: {destination_brain_path}")

def main_entry():
    if len(sys.argv) > 1 and sys.argv[1].strip() not in [".", "./", ""]:
        target_dir = os.path.abspath(sys.argv[1].strip())
    else:
        target_dir = os.getcwd()
        
    print(f"[SYSTEM] Initializing high-resolution trainer on: {target_dir}")
    trainer = HDFA_RepoTrainer()
    trainer.execute_one_shot_training(target_dir)

if __name__ == "__main__":
    main_entry()
