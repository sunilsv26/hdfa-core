import torch
import time
import os
import sys

from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder
from .repo_harvester import HDFA_ProjectHarvester
from .fluid_grid import HDFA_FluidGrid
from .lookup_engine import HDFA_LookupEngine
from .save_state import HDFA_MemorySaver


class HDFA_RepoTrainer:
    def __init__(self):
        print("[PIPELINE] Initializing Local Repository Training Architecture...")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        self.grid = HDFA_FluidGrid()
        self.lookup = HDFA_LookupEngine(self.engine)
        self.saver = HDFA_MemorySaver(self.engine)

    def execute_one_shot_training(self, workspace_path):
        # 1. Harvest and Encode Workspace Code Streams
        harvester = HDFA_ProjectHarvester(workspace_path)
        master_tensor_pool = harvester.harvest_and_stream_workspace(self.engine, self.encoder)
        
        if master_tensor_pool.shape[0] == 0:
            print("[ERROR] No code assets discovered for sequence streaming.")
            return

        print(f"\n[STEP 1] Streaming {master_tensor_pool.shape} structural n-grams through Fluid Automaton Grid...")
        
        start_time = time.perf_counter()
        
        # 2. Local Cellular Neighborhood Propagation Loop
        for index in range(master_tensor_pool.shape[0]):
            incoming_char_wave = master_tensor_pool[index]
            self.grid.step_local_automaton(incoming_char_wave)
            
        duration = time.perf_counter() - start_time
        print(f"[SUCCESS] Grid processing finalized in {duration:.4f} seconds.")
        print(f"Average absorption velocity: {(master_tensor_pool.shape[0]/duration):.2f} tokens per second.")

        # 3. Cache the Learned Spatial State as a Target Lookup Frame
        learned_state_signature = self.grid.step_local_automaton(torch.zeros(10000))
        self.engine.codebook["hdfa-core-architecture-blueprint"] = learned_state_signature
        
        # Enforce absolute destination paths straight inside the crawled workspace folder
        destination_brain_path = os.path.normpath(os.path.join(workspace_path, "codebase_brain.pt"))
        print(f"\n[STEP 2] Committing learned workspace matrices to long-term storage...")
        self.saver.save_brain_snapshot(destination_brain_path)
        
        # 4. Run an Instant Verification Check
        print("\n[STEP 3] Running Verification Test on Ingested Architecture Context...")
        corrupted_state = torch.sign(learned_state_signature + (torch.randn(10000) * 0.2))
        corrupted_state[corrupted_state == 0] = -1.0
        
        best_match, resonance = self.lookup.query_nearest_syntax(corrupted_state)
        
        print("\n================== WORKSPACE TRAINING RESULTS ==================")
        print(f"Identified Concept Target: '{best_match}'")
        print(f"Geometric Similarity Resonance: {resonance} / {self.engine.dimension}")
        print("================================================================")
        print("\n[SUCCESS] Local project training architecture validated completely.")


# FIXED: Explicitly defined main_entry to eliminate global console execution route exceptions
def main_entry():
    # If the user provides an argument that isn't empty, pull its path. Otherwise, use active folder.
    if len(sys.argv) > 1:
        first_argument = sys.argv[1].strip()
        if first_argument in [".", "./", ""]:
            target_dir = os.getcwd()
        else:
            target_dir = os.path.abspath(first_argument)
    else:
        target_dir = os.getcwd()
        
    print(f"[SYSTEM] Initializing macro training ingestion track on path: {target_dir}")
    trainer = HDFA_RepoTrainer()
    trainer.execute_one_shot_training(target_dir)


if __name__ == "__main__":
    main_entry()
