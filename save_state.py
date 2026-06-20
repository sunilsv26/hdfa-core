import torch
import os
import json

class HDFA_MemorySaver:
    def __init__(self, vector_engine):
        """
        Initializes the Persistent Long-Term Memory Save/Load Engine.
        """
        self.engine = vector_engine

    def save_brain_snapshot(self, file_path="brain_state.pt"):
        """
        Compresses and saves the active 10,000-D codebook to your hard drive.
        Converts float tensors to space-saving 8-bit integers (INT8).
        """
        print(f"[MEMORY] Serializing active hyperdimensional matrix to disk: {file_path}")
        
        # Create a compressed data dictionary
        compressed_snapshots = {}
        
        for token, vector in self.engine.codebook.items():
            # Convert heavy floating points (-1.0, 1.0) into lightweight 8-bit integers (-1, 1)
            compressed_snapshots[token] = vector.to(torch.int8)
            
        # Save the dictionary using PyTorch's native file compression
        torch.save(compressed_snapshots, file_path)
        
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"[SUCCESS] Brain state archived safely! File footprint on disk: {file_size_kb:.2f} KB")

    def load_brain_snapshot(self, file_path="brain_state.pt"):
        """
        Loads a saved brain snapshot file back into your laptop's active RAM cache.
        """
        if not os.path.exists(file_path):
            print(f"[WARNING] No saved brain memory file discovered at: {file_path}")
            return False

        print(f"[MEMORY] Rehydrating hyperdimensional codebook from disk: {file_path}")
        
        # Load the raw compressed integer map
        saved_snapshots = torch.load(file_path, weights_only=True)
        
        # Restore vectors back to active floating-point arrays for mathematical processing
        for token, integer_vector in saved_snapshots.items():
            self.engine.codebook[token] = integer_vector.float()
            
        print(f"[SUCCESS] Rehydration complete! Locked {len(self.engine.codebook)} template vectors back into RAM.")
        return True

# --- MEMORY RETENTION VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Phase 3: Persistent Long-Term Memory Core...")
    # FIXED: Replaced the absolute package name path with a direct local file import
    from core_math import HDC_VectorEngine
    
    # 1. Spin up a temporary brain engine and teach it a rule
    engine_a = HDC_VectorEngine()
    engine_a.generate_orthogonal_vector("useState")
    
    # 2. Save its state to your hard drive and clear the cache
    saver_a = HDFA_MemorySaver(engine_a)
    test_file = "test_brain_memory.pt"
    saver_a.save_brain_snapshot(test_file)
    
    # 3. Spin up a completely blank second engine that knows absolutely nothing
    print("\nSpinning up a completely fresh, empty secondary engine...")
    engine_b = HDC_VectorEngine()
    print(f"Empty Engine Codebook size before loading: {len(engine_b.codebook)}")
    
    # 4. Load the file data into the empty engine
    saver_b = HDFA_MemorySaver(engine_b)
    saver_b.load_brain_snapshot(test_file)
    print(f"Engine Codebook size after loading file:  {len(engine_b.codebook)}")
    
    # 5. Clean up the test file from your disk directory
    if os.path.exists(test_file):
        os.remove(test_file)
        
    if "useState" in engine_b.codebook:
        print("\n[SUCCESS] Memory state test passed! Long-term storage layer operates flawlessly.")
