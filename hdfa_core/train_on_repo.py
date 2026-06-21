import torch
import time
import os
import sys

from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder
from .save_state import HDFA_MemorySaver

class HDFA_RepoTrainer:
    def __init__(self):
        print("[PIPELINE] Initializing Matrix-Optimized Training Architecture...")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        self.saver = HDFA_MemorySaver(self.engine)

    def execute_one_shot_training(self, workspace_path):
        extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json']
        learned_vectors = []
        learned_strings = []
        processed_files_count = 0

        start_train = time.perf_counter()
        
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
                                if len(clean_line) > 10 and not clean_line.startswith(('import', 'from', '*', '//')):
                                    line_waves = self.encoder.encode_file_stream(clean_line)
                                    if line_waves is not None and line_waves.shape[0] > 0:
                                        line_vector = torch.sign(torch.sum(line_waves, dim=0))
                                        line_vector[line_vector == 0] = -1.0
                                        
                                        # PRODUCTION OPTIMIZATION: Convert vector to a highly compressed int8 byte format
                                        learned_vectors.append(line_vector.to(torch.int8).unsqueeze(0))
                                        learned_strings.append(clean_line)
                        processed_files_count += 1
                    except Exception:
                        continue

        if not learned_vectors:
            print("[ERROR] No unique code sequences discovered.")
            return

        train_duration = time.perf_counter() - start_train
        print(f"\n[SUCCESS] Compiled {len(learned_strings)} parallel code blocks in {train_duration:.2f}s!")

        # Stack separate arrays into a clean, optimized 2D matrix block
        master_matrix_block = torch.cat(learned_vectors, dim=0)
        
        # Package into a highly compact, optimized structural bundle
        packaged_brain_bundle = {
            "matrix_block": master_matrix_block,
            "text_strings": learned_strings
        }
        
        destination_brain_path = os.path.normpath(os.path.join(workspace_path, "codebase_brain.pt"))
        print(f"[STEP 2] Serializing compressed vector matrices to disk...")
        torch.save(packaged_brain_bundle, destination_brain_path)
        print(f"[SUCCESS] High-resolution brain snapshot sealed at: {destination_brain_path}")

def main_entry():
    target_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() not in [".", "./", ""] else os.getcwd()
    print(f"[SYSTEM] Initializing high-resolution trainer on: {target_dir}")
    trainer = HDFA_RepoTrainer()
    trainer.execute_one_shot_training(target_dir)

if __name__ == "__main__":
    main_entry()
