import torch
from core_math import HDC_VectorEngine

class HDFA_SlidingEncoder:
    def __init__(self, vector_engine, window_size=3):
        """
        Initializes the Character-Level Sequential Encoder.
        window_size: How many continuous characters are linked inside a time slice.
        """
        self.engine = vector_engine
        self.window_size = window_size

    def encode_file_stream(self, raw_text_content):
        """
        Transforms a continuous code script file into a sequence of bound 
        hyperdimensional structural waves by building n-grams from base characters.
        """
        sequence_hypervectors = []
        
        # Fallback if the input text is shorter than the target window window_size
        if len(raw_text_content) < self.window_size:
            fallback_vector = self.engine.generate_orthogonal_vector(raw_text_content)
            return torch.stack([fallback_vector])

        # Slide across the character timeline stream step-by-step
        for i in range(len(raw_text_content) - self.window_size + 1):
            window_slice = raw_text_content[i : i + self.window_size]
            
            # 1. Start with the base fingerprint of the absolute first character
            first_char = window_slice[0]
            bound_window_vector = self.engine.generate_orthogonal_vector(first_char)
            
            # 2. Extract, positionally shift, and bind the remaining characters sequentially
            for position, char in enumerate(window_slice[1:], start=1):
                char_vector = self.engine.generate_orthogonal_vector(char)
                
                # Cyclic shift (roll) provides sequence memory context
                # This guarantees Vector('c'+'o'+'n') != Vector('n'+'o'+'c')
                permuted_char_vector = torch.roll(char_vector, shifts=position, dims=0)
                
                # Element-wise multiplication (XOR logic) compiles them into an n-gram molecule
                bound_window_vector = bound_window_vector * permuted_char_vector
                
            sequence_hypervectors.append(bound_window_vector)
            
        return torch.stack(sequence_hypervectors)

# --- PHASE 2 ARCHITECTURE VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Phase 2: Sliding-Window Character Encoder Core...")
    base_engine = HDC_VectorEngine()
    encoder = HDFA_SlidingEncoder(base_engine, window_size=3)
    
    # Test texts: Verifying that sequence permutation logic isolates patterns
    code_string_a = "con"
    code_string_b = "noc"
    
    # Process sequence streams
    wave_a = encoder.encode_file_stream(code_string_a)
    wave_b = encoder.encode_file_stream(code_string_b)
    
    # Flatten the outputs to 1D vectors for a clean dot product resonance comparison
    vec_a = wave_a[0]
    vec_b = wave_b[0]
    
    # Measure overlap to ensure structural distinction
    sequence_resonance = base_engine.compute_orthogonality(vec_a, vec_b)
    
    print(f"Processed Window Block Space Matrix Shape: {wave_a.shape}")
    print(f"Sequence Resonance Score between '{code_string_a}' and '{code_string_b}': {sequence_resonance}")
    
    # Safe academic threshold accounting for normal 10,000-D statistical variance (3 standard deviations)
    if abs(sequence_resonance) < 300:
        print("\n[SUCCESS] Phase 2 validation passed. Identical letters in different positions generate unique orthogonal waves!")
    else:
        print("\n[ERROR] Sequence positioning overlap failure.")
