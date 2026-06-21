import asyncio
import json
import sys
import os
import torch
from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder

class HDFA_IDEMatrixChatServer:
    def __init__(self, host="127.0.0.1", port=8765):
        if len(sys.argv) > 1:
            try:
                port = int(sys.argv[1])
            except ValueError:
                pass
                
        self.host = host
        self.port = port
        print(f"[BRIDGE] Initializing matrix-vector sandbox engine on port {self.port}...")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        self.matrix_block = None
        self.text_strings = []
        
        # Load the newly optimized 2D matrix brain bundle from disk
        local_brain = os.path.normpath(os.path.join(os.getcwd(), "codebase_brain.pt"))
        if os.path.exists(local_brain):
            try:
                bundle = torch.load(local_brain, map_location=torch.device('cpu'))
                if isinstance(bundle, dict) and "matrix_block" in bundle:
                    self.matrix_block = bundle["matrix_block"].float() # Pre-cast to Float32 for math speed
                    self.text_strings = bundle["text_strings"]
                    print(f"[SUCCESS] Sandbox rehydrated {self.matrix_block.shape} lines in 2D parallel matrix tensor!")
            except Exception as e:
                print(f"[ERROR] Matrix brain rehydration failure: {str(e)}")
        else:
            print(f"[WARN] No codebase_brain.pt found inside working directory: {os.getcwd()}")
        sys.stdout.flush()

    async def handle_chat_stream(self, websocket):
        print(f"[BRIDGE] Active matrix chat session linked: {websocket.remote_address}")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                chat_prompt = data.get("chat_prompt", "").strip()
                if not chat_prompt or self.matrix_block is None:
                    continue
                
                print(f"[CHAT-QUERY] Ingesting prompt string: '{chat_prompt}'")
                    
                # 1. Project what the user typed into a 10,000-D query vector shape
                query_waves = self.encoder.encode_file_stream(chat_prompt)
                if query_waves is None or query_waves.shape == 0:
                    await websocket.send(json.dumps({"chat_response": "Could not project query vector."}))
                    continue
                    
                query_vector = torch.sign(torch.sum(query_waves, dim=0)).float()
                query_vector[query_vector == 0] = -1.0

                # 2. HIGH-SPEED MATRIX OPERATION
                # Runs a single vectorized parallel matrix-vector product step across all 8,000+ lines instantly
                scores = torch.mv(self.matrix_block, query_vector)
                
                # Boost match score if user prompt string matches a functional line keyword natively
                keywords = [word.lower() for word in chat_prompt.split() if len(word) > 2]
                for idx, raw_line_text in enumerate(self.text_strings):
                    line_lower = raw_line_text.lower()
                    if any(word in line_lower for word in ['state', 'auth', 'login', 'user', 'token', 'fetch']):
                        if any(kw in line_lower for kw in keywords):
                            scores[idx] += 3000.0

                # Extract the top 5 highest matching tensor indexes
                top_scores, top_indices = torch.topk(scores, min(5, len(self.text_strings)))

                # 3. Format response panel block
                response_text = f"🔍 [SANDBOX PORT {self.port}] Discovered parallel matrix context matches:\n\n"
                for rank, (score, index_tensor) in enumerate(zip(top_scores, top_indices), start=1):
                    idx = index_tensor.item()
                    line_text = self.text_strings[idx]
                    response_text += f"📍 Match {rank} (Resonance: {round(score.item(), 1)}):\n   ```javascript\n   {line_text}\n   ```\n\n"

                await websocket.send(json.dumps({"chat_response": response_text}))
                
            except Exception as e:
                print(f"[SANDBOX MATRIX EXCEPTION] {str(e)}")
                await websocket.send(json.dumps({"chat_response": f"🚨 Internal process breakdown: {str(e)}"}))
                break

    async def start_server(self):
        import websockets
        async with websockets.serve(self.handle_chat_stream, self.host, self.port, reuse_address=True):
            print(f"\n[SUCCESS] HDFA Chat Server running on port {self.port}")
            sys.stdout.flush()
            await asyncio.Future()

def main_bridge_entry():
    server = HDFA_IDEMatrixChatServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main_bridge_entry()
