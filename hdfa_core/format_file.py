import os
import torch
from hdfa_core.core_math import HDC_VectorEngine
from hdfa_core.sliding_encoder import HDFA_SlidingEncoder
from hdfa_core.cli import HDFA_IntegratedCLI

class HDFA_FileFormatter:
    def __init__(self, integrated_app):
        """
        Initializes the Full-File Auto-Correction System.
        """
        self.app = integrated_app

    def format_broken_script_file(self, input_file_path, output_file_path):
        """
        Reads a corrupted script file line-by-line, repairs the structure 
        using hyperdimensional trace alignment, and outputs a pristine clone.
        """
        if not os.path.exists(input_file_path):
            print(f"[ERROR] Target file source not found: {input_file_path}")
            return

        print(f"[FORMATTER] Reading corrupted asset source: {input_file_path}")
        repaired_lines = []

        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Process each individual line sequence through the HDFA alignment matrices
        for line_num, line_content in enumerate(lines, start=1):
            clean_line = line_content.strip()
            
            if not clean_line:
                repaired_lines.append("") # Keep empty spacing rows intact
                continue

            # Project current line characters into sequential hyperdimensional waves
            query_waves = self.app.encoder.encode_file_stream(clean_line)
            
            # Find the closest matching official documentation syntax block template
            matched_line, trace_score = self.app.query_sequence_alignment(query_waves)
            
            # If the trace score shows a high structural resonance, we fix the line!
            # Otherwise, we keep the original code line to avoid breaking user logic
            if trace_score > 3000.0:
                repaired_lines.append(matched_line)
                print(f" ├── [Line {line_num}] Repaired: '{clean_line}' ──> '{matched_line}' (Score: {trace_score:.1f})")
            else:
                repaired_lines.append(clean_line)

        # Write out the cleanroom compiled code back to disk
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(repaired_lines) + "\n")
            
        print(f"\n[SUCCESS] File synthesis complete! Healed codebase script saved to: {output_file_path}")

# --- FILE AUTOMATION ENGINE VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Phase 2: Full File Automated Repair Engine...")
    
    # 1. Spin up the master application core
    app_core = HDFA_IntegratedCLI()
    formatter = HDFA_FileFormatter(app_core)
    
    # 2. Create a simulated broken react file on your drive for testing
    simulated_broken_file = "broken_component.jsx"
    simulated_fixed_file = "fixed_component.jsx"
    
    broken_code_lines = [
        "const [state, setState] = useSt",  # Broken useState hook
        "",
        "useEffect(() => { fetchData();",   # Broken useEffect hook
        "",
        "return (<div><Component"           # Broken HTML layout tag
    ]
    
    with open(simulated_broken_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(broken_code_lines))
        
    # 3. Trigger the hyperdimensional file auto-repair formatter pipeline
    print("\n[START] Injecting corrupted script layout file into HDFA engine...")
    formatter.format_broken_script_file(simulated_broken_file, simulated_fixed_file)
    
    # 4. Clean up temporary evaluation files from your disk workspace
    if os.path.exists(simulated_broken_file):
        os.remove(simulated_broken_file)
