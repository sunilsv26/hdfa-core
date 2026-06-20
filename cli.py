import sys
import torch
from core_math import HDC_VectorEngine
from sliding_encoder import HDFA_SlidingEncoder
from predictor import HDFA_CharacterPredictor

class HDFA_IntegratedCLI:
    def __init__(self):
        print("================================================================")
        print("🧠 HDFA INTEGRATED CORE: AUTO-REPAIR & PREDICTIVE INTERFACE")
        print("================================================================")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        self.predictor = HDFA_CharacterPredictor(self.engine, self.encoder)
        
        # Seed both sequence repair memory and character transitions
        self._seed_reference_systems()

    def _seed_reference_systems(self):
        """Injects targeted reference templates into both memory tracks."""
        self.templates = [
            "const [state, setState] = useState(initial);",
            "useEffect(() => { fetchData(); }, []);",
            "return (<div><Component /></div>);",
            "display: flex; justify-content: center; align-items: center;",
            "export default function App() { return null; }"
        ]
        
        self.template_vectors = {}
        for template in self.templates:
            # 1. Compile full sequence track matrices for whole-line auto-repair
            self.template_vectors[template] = self.encoder.encode_file_stream(template)
            # 2. Feed the same templates into the character predictor transition matrix
            self.predictor.learn_transitions_from_text(template)
            
        print(f"\n[SYSTEM] System primed. Synchronized {len(self.templates)} foundational layouts.")

    def query_sequence_alignment(self, query_waves):
        """Compares sequence tracks using an accumulated trace match matrix."""
        best_match_template = None
        highest_cumulative_resonance = -float('inf')

        for template, target_waves in self.template_vectors.items():
            cumulative_resonance = 0.0
            for q_vec in query_waves:
                dot_products = torch.matmul(target_waves, q_vec)
                max_resonance = torch.max(dot_products).item()
                cumulative_resonance += max_resonance
            
            # FIXED: Divide by the sequence length scalar (number of text rows)
            normalized_score = cumulative_resonance / query_waves.shape[0]
            if normalized_score > highest_cumulative_resonance:
                highest_cumulative_resonance = normalized_score
                best_match_template = template

        return best_match_template, highest_cumulative_resonance

    def run_repl_loop(self):
        print("\nEnter code text to trigger simultaneous auto-repair and token predictions.")
        print("Type 'exit' or 'quit' to terminate the session.\n")
        
        while True:
            try:
                user_query = input("HDFA-Prompt >>> ")
                
                if not user_query.strip():
                    continue
                if user_query.strip().lower() in ['exit', 'quit']:
                    print("\n[INFO] Terminating session. Goodbye.")
                    sys.exit(0)

                # 1. Generate sequence tracking matrix
                query_waves = self.encoder.encode_file_stream(user_query)
                
                # 2. Line-Level Auto-Correction Task
                matched_line, line_score = self.query_sequence_alignment(query_waves)
                
                # 3. Next-Character Prediction Task
                next_char, char_resonance = self.predictor.predict_next_character(user_query)
                
                # Render results seamlessly to the prompt interface
                print(f" ├── 🛠️ Auto-Repair Suggestion: '{matched_line}' (Trace Fit: {line_score:.1f})")
                print(f" └── 🔮 Next Character Prediction: '{next_char}' (Synaptic Resonance: {char_resonance:.1f})\n")
                
            except KeyboardInterrupt:
                print("\n\n[INFO] Session terminated via hardware signal kill interrupt.")
                sys.exit(0)

if __name__ == "__main__":
    app = HDFA_IntegratedCLI()
    app.run_repl_loop()
