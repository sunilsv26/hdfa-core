import os
import torch
import sys
from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder

class HDFA_ProjectHarvester:
    def __init__(self, target_directory, file_extensions=None):
        """
        Initializes a production-grade, low-RAM streaming codebase crawler.
        """
        self.target_dir = target_directory
        self.extensions = file_extensions or ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.json']
        self.processed_files_count = 0

    def harvest_and_stream_workspace(self, vector_engine, sliding_encoder):
        """
        Streams and squashes incoming code matrices immediately into a singular
        10,000-D master architecture signature, keeping RAM overhead near zero.
        """
        print(f"\n[STREAMING-START] Scanning directory root space: {self.target_dir}")
        
        # PRODUCTION STEP: Allocate a single, fixed 10,000-D anchor vector inside RAM.
        # This acts as our master repository canvas block.
        master_repository_signature = torch.zeros(vector_engine.dimension)
        has_ingested_data = False

        # Step 1: Fast Directory Discovery Walk
        all_target_files = []
        for root, dirs, files in os.walk(self.target_dir):
            ignored_folders = ['node_modules', '.git', '__pycache__', '.pytest_cache', 'dist', 'build', '.next', 'out']
            dirs[:] = [d for d in dirs if d not in ignored_folders]
            
            for file in files:
                if file in ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'package.json']:
                    continue
                if any(file.endswith(ext) for ext in self.extensions):
                    all_target_files.append(os.path.join(root, file))
                    
        print(f"[STAGE-1-COMPLETE] Discovered {len(all_target_files)} target source files.")

        # Step 2: Stream, Add, and Instantly Free Memory Cache Rows
        print("\n[STAGE-2] Launching high-speed vector accumulation pipeline...")
        for index, file_path in enumerate(all_target_files, start=1):
            file_name = os.path.basename(file_path)
            print(f" -> [{index}/{len(all_target_files)}] TARGETING: {file_name}")
            sys.stdout.flush()
            
            try:
                if not os.path.exists(file_path):
                    continue
                    
                file_size = os.path.getsize(file_path)
                print(f"    | Step A: File size verified at {file_size} bytes.")
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read().strip()
                
                char_count = len(file_content)
                print(f"    | Step B: Successfully loaded {char_count} characters.")
                
                if char_count <= 20:
                    continue
                
                print(f"    | Step C: Computing sliding matrix window vectors...")
                sys.stdout.flush()
                
                # Encode file text characters into temporary local tensor grid blocks
                file_waves = sliding_encoder.encode_file_stream(file_content)
                print(f"    | Step D: Generation complete. Shape metrics: {file_waves.shape}")
                
                if file_waves is not None and file_waves.shape[0] > 0:
                    # PRODUCTION STEP: Squash the large multi-row file matrix into a single 10,000-D context trace
                    file_summary_vector = torch.sum(file_waves, dim=0)
                    
                    # Accumulate directly onto our master repository canvas block
                    master_repository_signature += file_summary_vector
                    has_ingested_data = True
                    self.processed_files_count += 1
                    print(f"    | Step E: [RAM-SAFE] Squashed context added to signature. Memory freed.")
                
                # FORCE GARBAGE COLLECTION TRACE: Instantly wipe local variables to free laptop RAM rows completely
                del file_waves
                
            except Exception as file_error:
                print(f"    | [SKIPPED] Operational bottleneck handled: {str(file_error)}")
                continue

        print(f"\n[SUCCESS] File crawling sweep complete! Synthesized {self.processed_files_count} script assets.")
        
        if has_ingested_data:
            # Normalize the master canvas signature back to strict stable binary switches (-1.0 or 1.0)
            final_binary_blueprint = torch.sign(master_repository_signature)
            final_binary_blueprint[final_binary_blueprint == 0] = -1.0
            
            # Reshape into a 2D matrix view to maintain strict downstream compatibility with train_on_repo.py
            return final_binary_blueprint.unsqueeze(0)
        else:
            print("[WARN] No clean code files were successfully aggregated.")
            return torch.zeros(0, vector_engine.dimension)
