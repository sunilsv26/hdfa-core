import asyncio
import json
import sys
import torch
from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder
from .predictor import HDFA_CharacterPredictor
from .cli import HDFA_IntegratedCLI

class HDFA_IDEBridgeServer:
    def __init__(self, host="127.0.0.1", port=8765):
        """
        Initializes a background async server to stream model responses straight to an IDE.
        """
        self.host = host
        self.port = port
        print("[BRIDGE] Bootstrapping Hyper-Space Projection Matrices for IDE Context...")
        self.app = HDFA_IntegratedCLI()
        
        # Sync predictor maps with foundational layout references
        for template in self.app.templates:
            self.app.predictor.learn_transitions_from_text(template)

    async def handle_editor_stream(self, websocket):
        """
        Listens to keypress data streams from an editor plugin, executes lookups 
        in microseconds, and routes structural fixes back instantly.
        """
        print(f"[BRIDGE] Active editor socket connection established via: {websocket.remote_address}")
        
        encoder = self.app.encoder
        predictor = self.app.predictor
        
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # FIXED: Aligned the key string handle to look for 'code_line' from extension.js
                user_code_line = data.get("code_line", "").strip()
                
                if not user_code_line:
                    continue
                
                # Print confirmation log so you can see your live keystrokes hitting the Python core!
                print(f"[STREAM] Ingesting typing slice: '{user_code_line}'")
                    
                # 1. Project editor string into sequential character n-grams
                query_waves = encoder.encode_file_stream(user_code_line)
                
                # 2. Compute timeline sequence trace matching for whole-line auto-repair
                matched_line, line_score = self.app.query_sequence_alignment(query_waves)
                
                # 3. Compute next-character predictive transition trace
                prediction_result = predictor.predict_next_character(user_code_line)
                if isinstance(prediction_result, tuple):
                    next_char, char_resonance = prediction_result
                else:
                    next_char, char_resonance = " ", 0.0

                # Balanced alignment gate: lower to 3500.0 to catch partial lines easily
                if line_score < 3500.0:
                    matched_line = "No Confident Match Found"
                    next_char, char_resonance = " ", 0.0

                # Pack results into a high-speed payload package
                response_payload = {
                    "auto_repair_suggestion": matched_line,
                    "trace_fit_score": round(line_score, 1),
                    "next_character_prediction": next_char,
                    "synaptic_force": round(char_resonance, 1)
                }
                
                # Transmit payload directly back onto the text editor extension canvas layer
                await websocket.send(json.dumps(response_payload))
                
            except Exception as e:
                print(f"[BRIDGE EXCEPTION] {str(e)}")
                break
                
        print(f"[BRIDGE] Editor socket session disconnected: {websocket.remote_address}")

    async def start_server(self):
        """Orchestrates the asynchronous background runtime loop."""
        import websockets
        async with websockets.serve(self.handle_editor_stream, self.host, self.port):
            print(f"\n[SUCCESS] HDFA Plugin Bridge running at ws://{self.host}:{self.port}")
            print("Listening for live editor connection threads... Press Ctrl+C to close.")
            await asyncio.Future()

def main_bridge_entry():
    server = HDFA_IDEBridgeServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n[INFO] Background bridge engine closed gracefully via system hardware interrupt.")
        sys.exit(0)

if __name__ == "__main__":
    main_bridge_entry()
