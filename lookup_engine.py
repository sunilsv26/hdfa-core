import torch

class HDFA_LookupEngine:
    def __init__(self, vector_engine):
        """
        Initializes the Cleanroom Retrieval Interface.
        Uses pure geometric dot products to compare incoming noisy signals 
        against the system's compiled clean documentation codebook.
        """
        self.engine = vector_engine

    def query_nearest_syntax(self, test_wave_vector):
        """
        Acts as the model's auto-correct. It loops through all known 
        code patterns in the codebook and finds the one that creates 
        the highest 'resonance' (cosine similarity/dot product score).
        """
        best_match_token = None
        highest_resonance = -float('inf')

        # Scan across every clean phrase learned from the official documentation
        for token, clean_vector in self.engine.codebook.items():
            # The Dot Product measures the precise alignment of 10,000 switches
            resonance_score = torch.dot(test_wave_vector, clean_vector).item()

            if resonance_score > highest_resonance:
                highest_resonance = resonance_score
                best_match_token = token

        return best_match_token, highest_resonance

# --- DAY 5 VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Day 5: Cleanroom Dot-Product Lookup Engine...")
    from core_math import HDC_VectorEngine
    
    # 1. Spin up the Day 1 vector core
    engine = HDC_VectorEngine()
    lookup_core = HDFA_LookupEngine(engine)
    
    # 2. Simulate learning multiple lines of documentation syntax
    print("\n[INFO] Injecting reference React documentation templates into Codebook...")
    target_1 = "const [text, setText] = useState('');"
    target_2 = "useEffect(() => { fetchData(); }, []);"
    target_3 = "return (<div><Component /></div>);"
    
    v1 = engine.generate_orthogonal_vector(target_1)
    v2 = engine.generate_orthogonal_vector(target_2)
    v3 = engine.generate_orthogonal_vector(target_3)
    
    # 3. Create a "corrupted/broken" code input (Simulating code with bugs/noise)
    print("[INFO] Creating a highly corrupted wave of Target 1 (50% background noise)...")
    random_noise = torch.randint(0, 2, (engine.dimension,)).float()
    random_noise[random_noise == 0] = -1.0
    
    # Mix the original clean vector with random noise to simulate a broken string
    corrupted_signal = torch.sign(v1 + (random_noise * 1.0))
    corrupted_signal[corrupted_signal == 0] = -1.0
    
    # 4. Trigger the Cleanroom Lookup to auto-correct the wave
    matched_code, resonance = lookup_core.query_nearest_syntax(corrupted_signal)
    
    print(f"\nEngine Retrieval Result: '{matched_code}'")
    print(f"Signal Resonance Score: {resonance} out of {engine.dimension}")
    
    if matched_code == target_1:
        print("\n[SUCCESS] Day 5 complete! The lookup engine successfully repaired the corrupted wave back to perfect syntax.")
    else:
        print("\n[ERROR] Resonance matching failed to isolate the correct template.")
