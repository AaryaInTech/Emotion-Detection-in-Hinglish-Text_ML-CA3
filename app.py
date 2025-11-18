import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page configuration
st.set_page_config(
    page_title="Hindi/Hinglish Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_LABELS = len(GOEMOTIONS)
MAX_LEN = 96
MODEL_NAME = "distilbert-base-multilingual-cased"

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load from saved model directory, otherwise load base model
    model_path = "models/distilbert-emotion"
    if os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=NUM_LABELS, problem_type="multi_label_classification"
            ).to(device)
            st.sidebar.success("✅ Loaded trained model from disk")
            return model, tokenizer, device
        except Exception as e:
            st.sidebar.warning(f"Could not load saved model: {e}. Using base model.")
    
    # Fallback to base model (will need training)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    ).to(device)
    st.sidebar.warning("⚠️ Using base model. Please train the model first using the notebook.")
    return model, tokenizer, device

def predict_emotions(text, model, tokenizer, device, threshold=0.5, top_k=None):
    """Predict emotions from text"""
    enc = tokenizer([text], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    if top_k:
        idx = np.argsort(probs)[-top_k:][::-1]
        return [(GOEMOTIONS[i], float(probs[i])) for i in idx]

    result = [(GOEMOTIONS[i], float(probs[i])) for i in range(len(probs)) if probs[i] >= threshold]
    result = sorted(result, key=lambda x: x[1], reverse=True)

    if not result:
        idx = int(np.argmax(probs))
        result = [(GOEMOTIONS[idx], float(probs[idx]))]

    return result, probs

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">😊 Hindi/Hinglish Emotion Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect emotions in Hindi and Hinglish text using DistilBERT</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        top_k = st.slider("Show Top K Emotions", 1, 10, 5)
        show_all = st.checkbox("Show All Emotions", False)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.info("""
        **Model:** DistilBERT Multilingual  
        **Labels:** 28 Emotions  
        **Task:** Multi-label Classification
        """)
        
        st.markdown("---")
        st.header("💡 Example Texts")
        examples = [
            "मुझे बहुत गुस्सा आ रहा है",
            "मैं खुश हूँ",
            "मैं बहुत परेशान हूँ, मुझे खेद है",
            "यह बहुत अच्छा है!",
            "मुझे डर लग रहा है"
        ]
        for i, ex in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.example_text = ex
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Text")
        # Text input
        default_text = st.session_state.get('example_text', '')
        text_input = st.text_area(
            "Type or paste your Hindi/Hinglish text here:",
            value=default_text,
            height=150,
            placeholder="Example: मुझे बहुत खुशी हो रही है!"
        )
        
        if 'example_text' in st.session_state:
            del st.session_state.example_text
    
    with col2:
        st.header("📈 Quick Stats")
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            st.metric("Words", word_count)
            st.metric("Characters", char_count)
        else:
            st.info("Enter text to see stats")
    
    # Predict button
    if st.button("🔍 Detect Emotions", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing emotions..."):
                emotions, all_probs = predict_emotions(text_input, model, tokenizer, device, threshold, top_k if not show_all else None)
            
            # Display results
            st.markdown("---")
            st.header("🎯 Detected Emotions")
            
            if emotions:
                # Create two columns for results
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    # Display emotions with progress bars
                    for emotion, prob in emotions:
                        st.markdown(f"**{emotion.capitalize()}**")
                        st.progress(prob)
                        st.caption(f"{prob:.1%} confidence")
                        st.markdown("---")
                
                with res_col2:
                    # Emotion distribution chart
                    emotion_df = pd.DataFrame(emotions, columns=['Emotion', 'Probability'])
                    fig = px.bar(
                        emotion_df, 
                        x='Probability', 
                        y='Emotion',
                        orientation='h',
                        title="Emotion Probabilities",
                        color='Probability',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show all emotions if requested
            if show_all:
                st.markdown("---")
                st.header("📊 All Emotion Scores")
                all_emotions_df = pd.DataFrame({
                    'Emotion': GOEMOTIONS,
                    'Probability': all_probs
                }).sort_values('Probability', ascending=False)
                
                # Heatmap visualization
                fig = px.bar(
                    all_emotions_df,
                    x='Emotion',
                    y='Probability',
                    title="All Emotion Probabilities",
                    color='Probability',
                    color_continuous_scale='Viridis'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(
                    all_emotions_df.style.background_gradient(subset=['Probability'], cmap='Blues'),
                    use_container_width=True,
                    height=400
                )
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with ❤️ using Streamlit and DistilBERT</p>
        <p>Supports Hindi and Hinglish text emotion detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

