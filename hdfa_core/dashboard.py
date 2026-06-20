import streamlit as st
import torch
import os
from pathlib import Path
from hdfa_core.core_math import HDC_VectorEngine
from hdfa_core.sliding_encoder import HDFA_SlidingEncoder
from hdfa_core.predictor import HDFA_CharacterPredictor
from hdfa_core.cli import HDFA_IntegratedCLI
from hdfa_core.save_state import HDFA_MemorySaver

# 1. Page Global UI Configuration Settings
st.set_page_config(page_title="HDFA Brain-Like Dashboard Core", layout="wide", page_icon="🧠")

st.title("🧠 Hyper-Dimensional Fluid Automaton (HDFA) Core Dashboard")
st.write("A real-time visualization workspace tracing ultra-low-energy code synthesis and localized cellular dynamics.")

# 2. Persist Engine Components in Application State Memory Cache
if "app" not in st.session_state:
    with st.spinner("Initializing Hyper-Space Projection Matrices..."):
        st.session_state.app = HDFA_IntegratedCLI()
        for template in st.session_state.app.templates:
            st.session_state.app.predictor.learn_transitions_from_text(template)

# Extract shared references from state handles
app = st.session_state.app
engine = app.engine
encoder = app.encoder
predictor = app.predictor
saver = HDFA_MemorySaver(engine)

# 3. Create Sidebar Control and Reference Deck panels
st.sidebar.header("📁 System Knowledge Base Index")

# Long-Term Storage Controller — resolve brain file relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Candidate locations to look for the serialized brain file
candidate_paths = [
    PROJECT_ROOT / "codebase_brain.pt",
    Path(__file__).resolve().parent / "codebase_brain.pt",
]

# Pick the first existing candidate path
found_path = None
for p in candidate_paths:
    if p.exists():
        found_path = p
        break

if found_path:
    # Auto-load the brain file once per session to avoid requiring manual button clicks
    if "brain_loaded" not in st.session_state:
        with st.sidebar.spinner("Auto-loading brain snapshot from disk..."):
            success = saver.load_brain_snapshot(str(found_path))
            if success:
                for template in app.templates:
                    app.template_vectors[template] = encoder.encode_file_stream(template)
                st.sidebar.success("Brain & Templates Synchronized Natively!")
            else:
                st.sidebar.warning("Failed to load brain snapshot automatically; try the Rehydrate button.")
        st.session_state.brain_loaded = True

    if st.sidebar.button(f"🔌 Rehydrate '{found_path.name}' ({found_path.stat().st_size/1024:.2f} KB)", type="primary"):
        with st.sidebar.spinner("Pumping matrix states to CPU cache..."):
            success = saver.load_brain_snapshot(str(found_path))
            if success:
                for template in app.templates:
                    app.template_vectors[template] = encoder.encode_file_stream(template)
                st.sidebar.success("Brain & Templates Synchronized Natively!")
else:
    st.sidebar.warning("No persistent memory asset discovered. Run 'train_on_repo.py' first.")

st.sidebar.write("Active structural templates locked inside Codebook:")
# Clean up display clutter by only showing non-single character keys
visible_tokens = [k for k in engine.codebook.keys() if len(k) > 4]
for token in visible_tokens[:15]:
    st.sidebar.caption(f"📍 {token}")

# 4. Interactive User Code Prompt Segment
st.subheader("⌨️ Live Code Prompt")
user_input = st.text_input("Type partial, noisy, or broken React / JS code syntax blocks here:", 
                           value="const [state, setState] = useSt")

if user_input:
    # 5. Core Mathematical Pipeline Processing Steps
    query_waves = encoder.encode_file_stream(user_input)
    
    best_match_template = "No Confident Match Found"
    highest_alignment_score = 0.0

    # Combine baseline items into structural target loops
    for template in app.templates:
        target_waves = app.template_vectors[template]

        cumulative_resonance = 0.0
        for q_vec in query_waves:
            dot_products = torch.matmul(target_waves, q_vec)
            max_resonance = torch.max(dot_products).item()
            cumulative_resonance += max_resonance
        
        # FIXED: Extract only the sequence dimension element [0] to compute standard scalar division
        normalized_alignment = cumulative_resonance / query_waves.shape[0]

        if normalized_alignment > highest_alignment_score and normalized_alignment > 4000.0:
            highest_alignment_score = normalized_alignment
            best_match_template = template

    # Next-Character Prediction Task
    prediction_result = predictor.predict_next_character(user_input)
    if isinstance(prediction_result, tuple):
        next_char, char_resonance = prediction_result
    else:
        next_char, char_resonance = " ", 0.0

    # Clean character predictive trace noise if line template matches fail
    if best_match_template == "No Confident Match Found":
        next_char, char_resonance = " ", 0.0

    # 6. Display Metric Analysis Cards Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🛠️ Auto-Repair Confidence Fit", value=f"{highest_alignment_score:.1f} / 10000")
    with col2:
        st.metric(label="🔮 Next Character Predicted", value=f"'{next_char}'")
    with col3:
        st.metric(label="⚡ Synaptic Transition Force", value=f"{char_resonance:.1f} / 10000")

    # 7. Render Healed Output Blocks Block
    st.subheader("✅ Structural Auto-Correction Result")
    if best_match_template == "No Confident Match Found":
        st.info("No Confident Match Found. Keep typing to narrow down the template context.")
    else:
        st.code(best_match_template, language="javascript")

    # 8. Visualizing the Fluid Cellular Automaton Grid Matrix
    st.subheader("🌊 2D Localized Fluid Grid Cellular Automaton Ripple Matrix")
    
    last_wave_vector = query_waves[-1]
    spatial_2d_grid = last_wave_vector.view(100, 100).detach().clone()
    normalized_pixels = ((spatial_2d_grid + 1.0) / 2.0 * 255.0).byte().numpy()
    st.image(normalized_pixels, caption="Active Binary State Fluctuations (-1 vs 1 Cells)", width="stretch")
