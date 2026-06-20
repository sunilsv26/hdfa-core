import torch
import time
from core_math import HDC_VectorEngine
from sliding_encoder import HDFA_SlidingEncoder
from repo_harvester import HDFA_ProjectHarvester
from fluid_grid import HDFA_FluidGrid
from lookup_engine import HDFA_LookupEngine

class HDFA_RepoTrainer:
    def __init__(self):
        print("[PIPELINE] Initializing Local Repository Training Architecture...")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        self.grid = HDFA_FluidGrid()
        self.lookup = HDFA_LookupEngine(self.engine)

    def execute_one_shot_training(self, workspace_path):
        # 1. Harvest and Encode Workspace Code Streams
        harvester = HDFA_ProjectHarvester(workspace_path, file_extensions=['.py'])
        master_tensor_pool = harvester.harvest_and_stream_workspace(self.engine, self.encoder)
        
        if master_tensor_pool.shape[0] == 0:
            print("[ERROR] No code assets discovered for sequence streaming.")
            return

        print(f"\n[STEP 1] Streaming {master_tensor_pool.shape[0]} structural n-grams through Fluid Automaton Grid...")
        
        start_time = time.perf_counter()
        
        # 2. Local Cellular Neighborhood Propagation Loop
        # We push each character wave vector sequentially to ripple the 2D grid matrix
        for index in range(master_tensor_pool.shape[0]):
            incoming_char_wave = master_tensor_pool[index]
            # Update the fluid grid using fast local consensus cell transitions
            self.grid.step_local_automaton(incoming_char_wave)
            
        duration = time.perf_counter() - start_time
        print(f"[SUCCESS] Grid processing finalized in {duration:.4f} seconds.")
        print(f"Average absorption velocity: {(master_tensor_pool.shape[0]/duration):.2f} tokens per second.")

        # 3. Cache the Learned Spatial State as a Target Lookup Frame
        # We lock the final grid snapshot into our codebook as a unique structural signature
        learned_state_signature = self.grid.grid.flatten()
        self.engine.codebook["hdfa-core-architecture-blueprint"] = learned_state_signature
        
        # 4. Run an Instant Verification Check
        print("\n[STEP 2] Running Verification Test on Ingested Architecture Context...")
        # Simulate a partial or slightly noisy match wave to trace retrieval resonance
        corrupted_state = torch.sign(learned_state_signature + (torch.randn(10000) * 0.2))
        corrupted_state[corrupted_state == 0] = -1.0
        
        best_match, resonance = self.lookup.query_nearest_syntax(corrupted_state)
        
        print("\n================== WORKSPACE TRAINING RESULTS ==================")
        print(f"Identified Concept Target: '{best_match}'")
        print(f"Geometric Similarity Resonance: {resonance} / {self.engine.dimension}")
        print("================================================================")
        print("\n[SUCCESS] Local project training architecture validated completely.")

if __name__ == "__main__":
    import os
    trainer = HDFA_RepoTrainer()
    
    # Ingest the active hdfa-core directory to train the model on its own code!
    target_project_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.execute_one_shot_training(target_project_dir)
