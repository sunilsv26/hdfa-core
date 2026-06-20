import torch
from core_math import HDC_VectorEngine
from sliding_encoder import HDFA_SlidingEncoder

class HDFA_CharacterPredictor:
    def __init__(self, vector_engine, sliding_encoder):
        """
        Initializes the Next-Character Transition Predictor Engine.
        """
        self.engine = vector_engine
        self.encoder = sliding_encoder
        self.transition_graph = {}

    def learn_transitions_from_text(self, raw_text):
        """
        Scans code text and records how character windows transition to the next letter.
        """
        if len(raw_text) <= self.encoder.window_size:
            return

        print(f"[PREDICTOR] Ingesting transition text maps ({len(raw_text)} chars)...")
        
        # Scan across text characters up to the second to last window point
        for i in range(len(raw_text) - self.encoder.window_size):
            context_window = raw_text[i : i + self.encoder.window_size]
            next_character = raw_text[i + self.encoder.window_size]
            
            # Generate the vector signature for the active sequence block window
            context_waves = self.encoder.encode_file_stream(context_window)
            context_vector = context_waves[-1] # Pull the compiled n-gram vector
            
            # Retrieve the destination character target fingerprint vector
            next_char_vector = self.engine.generate_orthogonal_vector(next_character)
            
            # Accumulate transition tensors using basic associative vector binding additions
            if context_window not in self.transition_graph:
                self.transition_graph[context_window] = torch.zeros(self.engine.dimension)
                
            self.transition_graph[context_window] += next_char_vector
            
        # Standardize all accumulated maps back to binary states (-1 or 1)
        for context in self.transition_graph:
            self.transition_graph[context] = torch.sign(self.transition_graph[context])
            self.transition_graph[context][self.transition_graph[context] == 0] = -1.0

    def predict_next_character(self, context_string, potential_candidates="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ();{}[]_=\'\" "):
        """
        Evaluates an input string window and guesses the absolute highest probability next character.
        """
        if len(context_string) < self.encoder.window_size:
            return " "
            
        # Isolate the exact trailing window chunk length
        target_context = context_string[-self.encoder.window_size:]
        
        if target_context not in self.transition_graph:
            return " " # Return blank space if context trace has never been encountered
            
        learned_transition_prediction_vector = self.transition_graph[target_context]
        
        best_candidate_char = None
        highest_transition_resonance = -float('inf')
        
        # Scan through alphabetical and punctuation candidate keys to read max dot product
        for candidate in potential_candidates:
            candidate_vector = self.engine.generate_orthogonal_vector(candidate)
            resonance = torch.dot(learned_transition_prediction_vector, candidate_vector).item()
            
            if resonance > highest_transition_resonance:
                highest_transition_resonance = resonance
                best_candidate_char = candidate
                
        return best_candidate_char, highest_transition_resonance

# --- PREDICTOR CORE VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Phase 2: N-Gram Predictive Token Generator Core...")
    base_engine = HDC_VectorEngine()
    base_encoder = HDFA_SlidingEncoder(base_engine, window_size=3)
    predictor = HDFA_CharacterPredictor(base_engine, base_encoder)
    
    # 1. Feed training boilerplate examples
    sample_source_code = "const [state, setState] = useState(null); useEffect(() => {}, []);"
    predictor.learn_transitions_from_text(sample_source_code)
    
    # 2. Test next-character predictive transitions
    test_prompt = "useSt" # The last 3 characters are 'eSt'
    predicted_letter, resonance = predictor.predict_next_character(test_prompt)
    
    print("\n================== CHARACTER PREDICTION RESULTS ==================")
    print(f"Context String Prompt: '{test_prompt}'")
    print(f"Predicted Next Letter: '{predicted_letter}' (Expected: 'a' from 'useState')")
    print(f"Synaptic Resonance:    {resonance:.1f} / {base_engine.dimension}")
    print("==================================================================")
    
    if predicted_letter == 'a':
        print("\n[SUCCESS] Predictor module validated. Character transitions mapped successfully.")
