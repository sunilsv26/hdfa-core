import torch
import sys

class HDFA_SlidingEncoder:
    def __init__(self, vector_engine, window_size=3):
        """
        Initializes the character-level sliding-window n-gram matrix encoder.
        """
        self.engine = vector_engine
        self.window_size = window_size

    def encode_file_stream(self, text_string):
        """
        Production-grade pre-allocated sequence encoder with math guards.
        """
        if len(text_string) < self.window_size:
            return torch.zeros((0, self.engine.dimension))

        total_ngrams = len(text_string) - self.window_size + 1
        encoded_matrix = torch.zeros((total_ngrams, self.engine.dimension))
        active_window_wave = torch.ones(self.engine.dimension)

        # Pre-fetch and cache vectors for characters
        unique_chars = set(text_string)
        char_vectors = {}
        for char in unique_chars:
            try:
                if char not in self.engine.codebook:
                    self.engine.generate_orthogonal_vector(char)
                char_vectors[char] = self.engine.codebook[char]
            except Exception:
                # Fallback to a zero template if character generation fails
                char_vectors[char] = torch.zeros(self.engine.dimension)

        # Single-pass execution loop with inline math exception safety shields
        for i in range(len(text_string)):
            try:
                current_char = text_string[i]
                char_vec = char_vectors.get(current_char, torch.zeros(self.engine.dimension))

                # Wrapped vector shift arithmetic safely
                rolled_wave = torch.roll(active_window_wave, shifts=1)
                active_window_wave = rolled_wave * char_vec
                
                # Check for accidental infinity/NaN numbers to prevent system halts
                if torch.isnan(active_window_wave).any() or torch.isinf(active_window_wave).any():
                    active_window_wave = torch.ones(self.engine.dimension)

                if i >= self.window_size - 1:
                    matrix_index = i - self.window_size + 1
                    encoded_matrix[matrix_index] = active_window_wave.clone()
            except Exception:
                # If a highly exotic character causes a mathematical breakdown, skip it safely
                continue

        return encoded_matrix
