import sys
import torch
from core_math import HDC_VectorEngine
from sliding_encoder import HDFA_SlidingEncoder

class HDFA_InteractiveCLI:
    def __init__(self):
        print("================================================================")
        print("🧠 HYPER-DIMENSIONAL FLUID AUTOMATON (HDFA) INTERACTIVE CLI CORE")
        print("================================================================")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        
        # We pre-populate the codebook with template targets
        self._seed_reference_codebook()

    def _seed_reference_codebook(self):
        """Injects targeted React, HTML, and CSS boilerplate references."""
        self.templates = [
            "const [state, setState] = useState(initial);",
            "useEffect(() => { fetchData(); }, []);",
            "return (<div><Component /></div>);",
            "display: flex; justify-content: center; align-items: center;",
            "export default function App() { return null; }"
        ]
        # Pre-compile the full sliding window sequences for our reference templates
        self.template_vectors = {}
        for template in self.templates:
            # This generates a [Length, 10000] matrix for each clean template
            self.template_vectors[template] = self.encoder.encode_file_stream(template)
            
        print(f"[SYSTEM] Compiled sequence tracking profiles for {len(self.templates)} templates.")

    def query_sequence_alignment(self, query_waves):
        """
        Compares sequence tracks using an accumulated trace match matrix.
        """
        best_match_template = None
        highest_cumulative_resonance = -float('inf')

        # Compare your user input waves against each pre-compiled template sequence track
        for template, target_waves in self.template_vectors.items():
            cumulative_resonance = 0.0
            
            # Measure resonance overlap for each sliding n-gram window step
            for q_vec in query_waves:
                # Find the maximum alignment for this query block inside the target template matrix
                # Max dot-product across the template dimension helps catch subset strings
                dot_products = torch.matmul(target_waves, q_vec)
                max_resonance = torch.max(dot_products).item()
                cumulative_resonance += max_resonance
            
            # Normalize score by the query size to evaluate relative accuracy
            normalized_score = cumulative_resonance / query_waves.shape[0]

            if normalized_score > highest_cumulative_resonance:
                highest_cumulative_resonance = normalized_score
                best_match_template = template

        return best_match_template, highest_cumulative_resonance

    def run_repl_loop(self):
        print("\nEnter a broken, noisy, or partial code string to trigger repair.")
        print("Type 'exit' or 'quit' to terminate the engine session.\n")
        
        while True:
            try:
                user_query = input("HDFA-Prompt >>> ").strip()
                
                if not user_query:
                    continue
                if user_query.lower() in ['exit', 'quit']:
                    print("\n[INFO] Terminating interactive context session. Goodbye.")
                    sys.exit(0)

                # 1. Process text inputs into sequential 3-character n-gram matrix frames [Length, 10000]
                query_waves = self.encoder.encode_file_stream(user_query)
                
                # 2. Compute timeline sequence trace alignment matching
                matched_syntax, trace_score = self.query_sequence_alignment(query_waves)
                
                # 3. Print out structural corrections instantly
                print(f" └─ Detected Repair Option: '{matched_syntax}'")
                print(f" └─ Normalized Sequence Trace Fit: {trace_score:.1f} / {self.engine.dimension}\n")
                
            except KeyboardInterrupt:
                print("\n\n[INFO] Session interrupted via hardware keyboard kill trigger. Exiting.")
                sys.exit(0)

if __name__ == "__main__":
    cli_app = HDFA_InteractiveCLI()
    cli_app.run_repl_loop()
