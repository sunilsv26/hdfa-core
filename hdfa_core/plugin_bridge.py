import asyncio
import json
import sys
import os
import torch
from .core_math import HDC_VectorEngine
from .sliding_encoder import HDFA_SlidingEncoder

class HDFA_IDEChatServer:
    def __init__(self, host="127.0.0.1", port=8765):
        # FIXED: Accept dynamic port overrides directly from runtime system argument vectors
        if len(sys.argv) > 1:
            try:
                port = int(sys.argv[1])
            except ValueError:
                pass
                
        self.host = host
        self.port = port
        print(f"[BRIDGE] Initializing isolated sandbox engine on port {self.port}...")
        self.engine = HDC_VectorEngine()
        self.encoder = HDFA_SlidingEncoder(self.engine, window_size=3)
        
        # Load the brain snapshot strictly inside the active current working directory path
        local_brain = os.path.normpath(os.path.join(os.getcwd(), "codebase_brain.pt"))
        if os.path.exists(local_brain):
            try:
                loaded_state = torch.load(local_brain, map_location=torch.device('cpu'))
                if isinstance(loaded_state, dict):
                    for key, val in loaded_state.items():
                        if len(str(key).strip()) > 8:
                            self.engine.codebook[str(key).strip()] = val
                print(f"[SUCCESS] Sandbox rehydrated context from local file: {local_brain}")
            except Exception as e:
                print(f"[ERROR] Local snapshot load failure: {str(e)}")
        else:
            print(f"[WARN] No codebase_brain.pt discovered inside current working directory: {os.getcwd()}")
        sys.stdout.flush()

    async def handle_chat_stream(self, websocket):
        print(f"[BRIDGE] Active sandboxed chat session linked: {websocket.remote_address}")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                chat_prompt = data.get("chat_prompt", "").strip()
                if not chat_prompt:
                    continue
                
                query_waves = self.encoder.encode_file_stream(chat_prompt)
                if query_waves is None or query_waves.shape == 0:
                    await websocket.send(json.dumps({"chat_response": "Could not project query vector."}))
                    continue
                    
                query_vector = torch.sign(torch.sum(query_waves, dim=0)).float()
                query_vector[query_vector == 0] = -1.0

                match_candidates = []
                keywords = [word.lower() for word in chat_prompt.split() if len(word) > 2]

                for raw_line_text, stored_vector in self.engine.codebook.items():
                    if isinstance(stored_vector, torch.Tensor) and len(stored_vector.shape) > 0:
                        line_lower = raw_line_text.lower()
                        if len(raw_line_text) <= 8:
                            continue

                        has_keyword_match = any(kw in line_lower for kw in keywords) if keywords else True
                        if has_keyword_match or len(keywords) == 0:
                            target_vector = stored_vector.float()
                            similarity = torch.dot(query_vector, target_vector).item()
                            
                            if any(word in line_lower for word in ['state', 'auth', 'login', 'user', 'token', 'fetch']):
                                if any(kw in line_lower for kw in keywords):
                                    similarity += 3000.0
                                
                            match_candidates.append((similarity, raw_line_text))

                match_candidates.sort(key=lambda x: x, reverse=True)
                top_matches = match_candidates[:5]

                if len(top_matches) > 0:
                    response_text = f"🔍 [SANDBOX PORT {self.port}] Discovered matches from local folder:\n\n"
                    for idx, (sim, line_text) in enumerate(top_matches, start=1):
                        response_text += f"📍 Match {idx}:\n   ```javascript\n   {line_text}\n   ```\n\n"
                else:
                    response_text = "🤷 No close vector code lines discovered for that question inside this folder workspace."

                await websocket.send(json.dumps({"chat_response": response_text}))
                
            except Exception as e:
                print(f"[SANDBOX EXCEPTION] {str(e)}")
                break

    async def start_server(self):
        import websockets
        async with websockets.serve(self.handle_chat_stream, self.host, self.port, reuse_address=True):
            print(f"\n[SUCCESS] HDFA Chat Server running on port {self.port}")
            sys.stdout.flush()
            await asyncio.Future()

def main_bridge_entry():
    server = HDFA_IDEChatServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main_bridge_entry()
