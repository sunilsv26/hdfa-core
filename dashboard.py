import streamlit as st
import torch
from core_math import HDC_VectorEngine
from sliding_encoder import HDFA_SlidingEncoder
from predictor import HDFA_CharacterPredictor
from cli import HDFA_IntegratedCLI

# 1. Page Global UI Configuration Settings
st.set_page_config(page_title="HDFA Brain-Like Dashboard Core", layout="wide", page_icon="🧠")

st.title("🧠 Hyper-Dimensional Fluid Automaton (HDFA) Core Dashboard")
st.write("A real-time visualization workspace tracing ultra-low-energy code synthesis and localized cellular dynamics.")

# 2. Persist Engine Components in Application State Memory Cache
if "app" not in st.session_state:
    with st.spinner("Initializing Hyper-Space Projection Matrices..."):
        st.session_state.app = HDFA_IntegratedCLI()
        # Feed baseline templates into the character predictor transition matrix
        for template in st.session_state.app.templates:
            st.session_state.app.predictor.learn_transitions_from_text(template)

# Extract shared references from state handles
app = st.session_state.app
engine = app.engine
encoder = app.encoder
predictor = app.predictor

# 3. Create Sidebar Control and Reference Deck panels
st.sidebar.header("📁 System Knowledge Base Index")
st.sidebar.write("Active structural templates locked inside Codebook:")
for template in app.templates:
    st.sidebar.code(template, language="javascript")

# 4. Interactive User Code Prompt Segment
st.subheader("⌨️ Live Code Prompt")
user_input = st.text_input("Type partial, noisy, or broken React / JS code syntax blocks here:", 
                           value="const [state, setState] = useSt")

if user_input:
    # 5. Core Mathematical Pipeline Processing Steps
    query_waves = encoder.encode_file_stream(user_input)
    matched_line, trace_score = app.query_sequence_alignment(query_waves)
    
    # FIXED: Run safe conditional unpacking logic to catch unseen strings like 'useMe'
    prediction_result = predictor.predict_next_character(user_input)
    if isinstance(prediction_result, tuple):
        next_char, char_resonance = prediction_result
    else:
        # Fallback values if the sequence context has never been encountered before
        next_char, char_resonance = " ", 0.0

    # 6. Display Metric Analysis Cards Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🛠️ Auto-Repair Confidence Fit", value=f"{trace_score:.1f} / 10000")
    with col2:
        st.metric(label="🔮 Next Character Predicted", value=f"'{next_char}'")
    with col3:
        st.metric(label="⚡ Synaptic Transition Force", value=f"{char_resonance:.1f} / 10000")

    # 7. Render Healed Output Blocks Block
    st.subheader("✅ Structural Auto-Correction Result")
    st.code(matched_line, language="javascript")

    # 8. Visualizing the Fluid Cellular Automaton Grid Matrix
    st.subheader("🌊 2D Localized Fluid Grid Cellular Automaton Ripple Matrix")
    st.write("Visualizing the wave mechanics as character vectors step across the array topology:")

    # Compress the trailing query wave vector back to its 100x100 2D cellular visualization frame
    last_wave_vector = query_waves[-1]
    spatial_2d_grid = last_wave_vector.view(100, 100).detach().clone()
    
    # Map values from [-1, 1] to standard image pixel coordinates for visual render
    normalized_pixels = ((spatial_2d_grid + 1.0) / 2.0 * 255.0).byte().numpy()
    
    # FIXED: Updated layout rendering configuration parameter string to comply with late-2025 width changes
    st.image(normalized_pixels, caption="Active Binary State Fluctuations (-1 vs 1 Cells)", width="stretch")
