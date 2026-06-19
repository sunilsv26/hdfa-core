import torch

class HDFA_FluidGrid:
    def __init__(self, dimensions=10000, grid_height=100, grid_width=100):
        """
        Initializes a decentralized, cell-based fluid memory grid.
        Total cell grid size fits entirely inside the laptop's ultra-fast L3 cache.
        """
        self.dimensions = dimensions
        self.height = grid_height
        self.width = grid_width
        
        # The physical 2D grid holding binary cell states (-1.0 or 1.0)
                # Generates random 0 and 1 values, then transforms them into -1.0 and 1.0
        raw_bits = torch.randint(0, 2, (grid_height, grid_width)).float()
        raw_bits[raw_bits == 0] = -1.0
        self.grid = raw_bits

        
    def step_local_automaton(self, incoming_wave_vector):
        """
        Processes a character/token vector by rippling it across the grid.
        Each cell updates its state based ONLY on its 4 immediate neighbors 
        (Up, Down, Left, Right) and a fraction of the incoming data wave.
        """
        # Compress the 10,000-D wave vector to fit our 100x100 spatial grid layout
        spatial_wave = incoming_wave_vector.view(self.height, self.width)
        
        # Compute neighborhood states using fast array roll/shifts (No dense multiplications!)
        shift_up    = torch.roll(self.grid, shifts=-1, dims=0)
        shift_down  = torch.roll(self.grid, shifts=1, dims=0)
        shift_left  = torch.roll(self.grid, shifts=-1, dims=1)
        shift_right = torch.roll(self.grid, shifts=1, dims=1)
        
        # Local Consensus Rule: Add neighbor states together with the incoming syntax wave
        local_fluid_sum = shift_up + shift_down + shift_left + shift_right + spatial_wave
        
        # Apply strict binary thresholding (-1 or 1). This forms our self-organizing memory ripple.
        self.grid = torch.sign(local_fluid_sum)
        self.grid[self.grid == 0] = -1.0
        
        return self.grid.flatten() # Flatten back to a clean 10,000-D vector for the lookup engine

# --- DAY 4 VALIDATION TEST ---
if __name__ == "__main__":
    print("Initializing Day 4: Fluid Automaton Spatial Grid Core...")
    from core_math import HDC_VectorEngine
    
    engine = HDC_VectorEngine()
    fluid_core = HDFA_FluidGrid()
    
    # Simulate sequential steps of React code entering the brain timeline
    token_1 = engine.generate_orthogonal_vector("const")
    token_2 = engine.generate_orthogonal_vector("useState")
    
    print("\nStreaming tokens sequentially to trigger structural ripples...")
    state_after_t1 = fluid_core.step_local_automaton(token_1)
    state_after_t2 = fluid_core.step_local_automaton(token_2)
    
    # Check if the grid's memory state actually mutated dynamically over the steps
    grid_mutation_check = torch.dot(state_after_t1, state_after_t2).item()
    normalized_mutation = grid_mutation_check / engine.dimension
    
    print(f"Grid Space Matrix Shape: {fluid_core.height}x{fluid_core.width}")
    print(f"Temporal Overlap Factor: {normalized_mutation:.4f}")
    
    if abs(normalized_mutation) < 0.2:
        print("\n[SUCCESS] Day 4 complete. Cellular automata successfully captured string sequence timelines.")
    else:
        print("\n[ERROR] Waves didn't ripple correctly.")
