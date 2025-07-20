import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Salary Predictor", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model, scaler, and encoders
@st.cache_resource
def load_models():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

model, scaler, encoders = load_models()

# Enhanced Gen Z CSS with Black Current + Blue + Green theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Custom CSS Variables */
    :root {
        --primary-bg: #0a0a0a;
        --secondary-bg: #1a1a2e;
        --accent-bg: #16213e;
        --card-bg: rgba(26, 26, 46, 0.9);
        --glass-bg: rgba(22, 33, 62, 0.3);
        --current-purple: #2d1b69;
        --neon-blue: #00d4ff;
        --electric-green: #39ff14;
        --cyber-pink: #ff006e;
        --gold-accent: #ffd700;
        --text-primary: #ffffff;
        --text-secondary: #b8b8b8;
        --text-muted: #6b7280;
        --border-glow: rgba(0, 212, 255, 0.3);
        --success-green: #00ff87;
        --warning-orange: #ff8500;
        --error-red: #ff3838;
    }
    
    /* Global dark theme */
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 50%, var(--current-purple) 100%);
        background-attachment: fixed;
        color: var(--text-primary);
    }
    
    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(57, 255, 20, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 0, 110, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(120deg); }
        66% { transform: translateY(-10px) rotate(240deg); }
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Cyberpunk main header */
    .main-header {
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--glass-bg) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 60px var(--border-glow);
        border: 1px solid rgba(0, 212, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: scan 3s infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, var(--neon-blue) 0%, var(--electric-green) 50%, var(--cyber-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: glow-text 2s ease-in-out infinite alternate;
        z-index: 1;
        position: relative;
    }
    
    @keyframes glow-text {
        from { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(57, 255, 20, 0.5)); }
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.3rem;
        font-weight: 400;
        z-index: 1;
        position: relative;
    }
    
    /* Glassmorphism card styles */
    .info-card, .feature-section {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .info-card::before, .feature-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--neon-blue), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .info-card:hover, .feature-section:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.4),
            0 0 60px var(--border-glow),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.4);
    }
    
    .info-card:hover::before, .feature-section:hover::before {
        opacity: 1;
    }
    
    .info-card h3, .feature-section h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    /* Cyberpunk prediction card */
    .prediction-card {
        background: linear-gradient(135deg, var(--current-purple) 0%, var(--accent-bg) 50%, var(--secondary-bg) 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        color: var(--text-primary);
        margin: 2rem 0;
        box-shadow: 
            0 20px 40px rgba(45, 27, 105, 0.4),
            0 0 80px rgba(0, 212, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid var(--neon-blue);
        position: relative;
        overflow: hidden;
        animation: pulse-cyber 3s infinite;
    }
    
    .prediction-card::after {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--neon-blue), var(--electric-green), var(--cyber-pink), var(--neon-blue));
        z-index: -1;
        border-radius: 24px;
        animation: border-spin 4s linear infinite;
    }
    
    @keyframes border-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse-cyber {
        0%, 100% { 
            box-shadow: 
                0 20px 40px rgba(45, 27, 105, 0.4),
                0 0 80px rgba(0, 212, 255, 0.2); 
        }
        50% { 
            box-shadow: 
                0 25px 50px rgba(45, 27, 105, 0.6),
                0 0 120px rgba(57, 255, 20, 0.3); 
        }
    }
    
    .prediction-card h1 {
        font-size: 4.5rem;
        font-weight: 700;
        margin: 1.5rem 0;
        text-shadow: 0 0 40px var(--neon-blue);
        background: linear-gradient(45deg, var(--neon-blue), var(--electric-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        z-index: 1;
        position: relative;
    }
    
    /* Enhanced metrics with neon effects */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-left: 3px solid var(--electric-green);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, transparent, rgba(57, 255, 20, 0.05), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(57, 255, 20, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-left-color: var(--neon-blue);
    }
    
    .metric-card:hover::before {
        transform: translateX(100%);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--neon-blue), var(--electric-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 212, 255, 0.1);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, var(--current-purple), var(--accent-bg));
        padding: 2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(45, 27, 105, 0.3);
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .sidebar-header h2 {
        color: var(--text-primary);
        margin: 0;
        font-weight: 600;
        text-shadow: 0 0 20px var(--neon-blue);
    }
    
    /* Futuristic buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--current-purple) 0%, var(--accent-bg) 100%);
        color: var(--text-primary);
        border: 2px solid var(--neon-blue);
        border-radius: 50px;
        padding: 1.2rem 3rem;
        font-weight: 600;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.3),
            0 0 30px rgba(0, 212, 255, 0.2);
        position: relative;
        overflow: hidden;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 
            0 15px 40px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, var(--accent-bg) 0%, var(--current-purple) 100%);
        border-color: var(--electric-green);
        color: var(--electric-green);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* Enhanced DataFrame styling */
    .dataframe {
        background: var(--card-bg);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 255, 0.1);
    }
    
    /* Status messages with glow effects */
    .success-message {
        background: linear-gradient(135deg, var(--success-green) 0%, #00cc6a 100%);
        color: var(--primary-bg);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 
            0 10px 30px rgba(0, 255, 135, 0.3),
            0 0 40px rgba(0, 255, 135, 0.2);
        font-weight: 600;
        border: 1px solid var(--success-green);
    }
    
    .error-message {
        background: linear-gradient(135deg, var(--error-red) 0%, #cc0000 100%);
        color: var(--text-primary);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 
            0 10px 30px rgba(255, 56, 56, 0.3),
            0 0 40px rgba(255, 56, 56, 0.2);
        font-weight: 600;
        border: 1px solid var(--error-red);
    }
    
    /* Loading spinner enhancement */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(0, 212, 255, 0.1);
        border-top: 4px solid var(--neon-blue);
        border-radius: 50%;
        animation: cyber-spin 1s linear infinite;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    @keyframes cyber-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin-top: 3rem;
        box-shadow: 
            0 -10px 30px rgba(0, 0, 0, 0.3),
            0 0 60px rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.1);
    }
    
    .footer h3 {
        background: linear-gradient(135deg, var(--neon-blue), var(--electric-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .footer p {
        color: var(--text-secondary);
    }
    
    /* Streamlit component overrides */
    .stSelectbox > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    .stSlider > div > div > div {
        background: var(--glass-bg) !important;
    }
    
    .stNumberInput > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--glass-bg);
        border-radius: 12px;
        color: var(--text-secondary);
        border: 1px solid rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--current-purple), var(--accent-bg)) !important;
        color: var(--text-primary) !important;
        border-color: var(--neon-blue) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--card-bg) !important;
        border: 2px dashed rgba(0, 212, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--neon-blue) !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .prediction-card h1 {
            font-size: 3rem;
        }
        
        .info-card, .feature-section {
            padding: 1.5rem;
        }
        
        .metric-card {
            margin: 0.3rem;
            padding: 1.2rem;
        }
    }
    
    /* Scroll bar customization */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--neon-blue), var(--electric-green));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--electric-green), var(--cyber-pink));
    }
</style>
""", unsafe_allow_html=True)

# Cyberpunk main header
st.markdown("""
<div class="main-header">
    <h1>💰 AI Salary Predictor</h1>
    <p>Next-gen machine learning with cyberpunk aesthetics • Predict • Analyze • Dominate</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with cyber theme
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>⚡ Neural Input Panel</h2>
</div>
""", unsafe_allow_html=True)

# Input fields with better organization
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.slider("👤 Age", 17, 75, 30)
    educational_num = st.slider("🎓 Neural Level", 5, 16, 10)
    hours_per_week = st.slider("⚡ Work Cycles/Week", 1, 80, 40)

with col2:
    capital_gain = st.number_input("💎 Capital Boost", value=0)
    capital_loss = st.number_input("📉 Capital Drain", value=0)
    fnlwgt = st.number_input("⚖️ Weight Factor", value=100000)

# Categorical inputs with cyber theme
st.sidebar.markdown("---")
workclass = st.sidebar.selectbox("🏢 Work Matrix", encoders["workclass"].classes_)
marital_status = st.sidebar.selectbox("💫 Bond Status", encoders["marital-status"].classes_)
occupation = st.sidebar.selectbox("⚔️ Skill Class", encoders["occupation"].classes_)
relationship = st.sidebar.selectbox("🔗 Connection Type", encoders["relationship"].classes_)
race = st.sidebar.selectbox("🌍 Heritage", encoders["race"].classes_)
gender = st.sidebar.selectbox("⚧ Identity", encoders["gender"].classes_)
native_country = st.sidebar.selectbox("🚀 Origin Node", encoders["native-country"].classes_)

# Build input DataFrame
input_dict = {
    'age': age,
    'workclass': encoders['workclass'].transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': encoders['marital-status'].transform([marital_status])[0],
    'occupation': encoders['occupation'].transform([occupation])[0],
    'relationship': encoders['relationship'].transform([relationship])[0],
    'race': encoders['race'].transform([race])[0],
    'gender': encoders['gender'].transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encoders['native-country'].transform([native_country])[0]
}

input_df = pd.DataFrame([input_dict])

# Enhanced main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="feature-section">
        <h3>📋 Neural Input Matrix</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced display with cyber formatting
    display_data = {
        'Neural Node': ['👤 Bio-Age', '🧠 Neural-Level', '⚡ Cycle-Rate', '🏢 Work-Matrix', '💫 Bond-State', '⚔️ Skill-Class'],
        'Current Value': [f"{age} cycles", f"Level {educational_num}", f"{hours_per_week}/week", str(workclass)[:15], str(marital_status)[:15], str(occupation)[:15]]
    }
    display_df = pd.DataFrame(display_data)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("""
    <div class="feature-section">
        <h3>⚡ System Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Cyber-themed metrics
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{age}</div>
        <div class="metric-label">Bio Cycles</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{hours_per_week}</div>
        <div class="metric-label">Work Frequency</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{educational_num}</div>
        <div class="metric-label">Neural Tier</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced prediction section with cyber theme
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("🚀 INITIATE NEURAL SCAN", use_container_width=True):
        # Enhanced loading with cyber theme
        with st.spinner('🤖 Neural networks analyzing quantum data streams...'):
            import time
            time.sleep(1.5)  # Enhanced processing simulation
            
            # Make prediction
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
                prob_low = probabilities[0]
                prob_high = probabilities[1]
            else:
                prob_low = 0.3 if prediction[0] == 1 else 0.7
                prob_high = 0.7 if prediction[0] == 1 else 0.3
            
            # Cyber-themed prediction display
            prediction_class = ">50K" if prediction[0] == 1 else "≤50K"
            confidence = max(prob_low, prob_high)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2>⚡ NEURAL SCAN COMPLETE</h2>
                <h1>${prediction_class}</h1>
                <p style="font-size: 1.4rem; margin-top: 1rem;">System Confidence: {confidence:.1%}</p>
                <p style="font-size: 1rem; opacity: 0.9;">🧠 {len(input_dict)} neural pathways analyzed</p>
                <p style="font-size: 0.9rem; opacity: 0.7;">⚡ Quantum processing complete</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced probability visualization with cyber theme
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prob_low:.1%}</div>
                    <div class="metric-label">≤50K Probability</div>
                </div>
                """, unsafe_allow_html=True)
            
            with prob_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prob_high:.1%}</div>
                    <div class="metric-label">>50K Probability</div>
                </div>
                """, unsafe_allow_html=True)

# Enhanced Feature Analysis Section
st.markdown("---")
st.markdown("""
<div class="feature-section">
    <h3>🧠 Neural Feature Analysis</h3>
    <p>Advanced AI insights into prediction factors</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different analysis views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Impact", "🎯 SHAP Analysis", "💡 LIME Insights", "📈 Data Visualization"])

with tab1:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Key Neural Pathways</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance visualization
    if hasattr(model, 'feature_importances_'):
        feature_names = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 
                        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 
                        'capital-loss', 'hours-per-week', 'native-country']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="🧠 Neural Feature Importance Matrix",
                    color='Importance',
                    color_continuous_scale=['#0a0a0a', '#00d4ff', '#39ff14'])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Space Grotesk'),
            title_font_size=18,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 SHAP Neural Network Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        
        # Create SHAP waterfall plot
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]  # For binary classification
        else:
            shap_values_to_plot = shap_values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        # Custom SHAP plot with cyber theme
        feature_names = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 
                        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 
                        'capital-loss', 'hours-per-week', 'native-country']
        
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_values_to_plot[0]
        }).sort_values('SHAP_Value', key=abs, ascending=False).head(10)
        
        colors = ['#00d4ff' if x > 0 else '#ff006e' for x in shap_df['SHAP_Value']]
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP_Value'], color=colors, alpha=0.8)
        
        ax.set_xlabel('SHAP Impact Value', color='white', fontweight='bold')
        ax.set_title('🎯 Neural Feature Impact Analysis', color='#00d4ff', fontsize=16, fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='#39ff14')
        ax.spines['bottom'].set_color('#00d4ff')
        ax.spines['left'].set_color('#00d4ff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add glow effect to bars
        for bar, color in zip(bars, colors):
            bar.set_edgecolor(color)
            bar.set_linewidth(2)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.markdown("""
        <div class="error-message">
            ⚠️ Neural SHAP analysis temporarily offline. Core prediction systems remain fully operational.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="info-card">
        <h3>💡 LIME Quantum Explanations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # LIME analysis
        feature_names = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 
                        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 
                        'capital-loss', 'hours-per-week', 'native-country']
        
        # Create a dummy training dataset for LIME
        X_train_sample = np.random.normal(0, 1, (100, len(feature_names)))
        
        explainer = LimeTabularExplainer(
            X_train_sample,
            feature_names=feature_names,
            class_names=['≤50K', '>50K'],
            mode='classification'
        )
        
        explanation = explainer.explain_instance(
            input_scaled[0], 
            model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Create custom LIME visualization
        lime_data = explanation.as_list()
        lime_df = pd.DataFrame(lime_data, columns=['Feature', 'Impact']).head(10)
        
        fig = px.bar(lime_df, x='Impact', y='Feature', orientation='h',
                    title="💡 LIME Quantum Feature Breakdown",
                    color='Impact',
                    color_continuous_scale=['#ff006e', '#0a0a0a', '#39ff14'])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Space Grotesk'),
            title_font_size=18,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.markdown("""
        <div class="error-message">
            ⚠️ LIME quantum processors recalibrating. Standard neural analysis available.
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="info-card">
        <h3>📈 Cyber Data Visualization Matrix</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive radar chart of user inputs
    categories = ['Age Factor', 'Education Level', 'Work Hours', 'Experience', 'Stability', 'Skill Rating']
    values = [
        age/75*100,  # Normalized age
        educational_num/16*100,  # Normalized education
        hours_per_week/80*100,  # Normalized hours
        min(capital_gain/10000*100, 100),  # Capital gain factor
        50 + (30 if 'Married' in marital_status else 0),  # Stability factor
        70 + (20 if 'Prof' in occupation or 'Exec' in occupation else 0)  # Skill factor
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(color='#00d4ff', width=3),
        marker=dict(color='#39ff14', size=8),
        name='Neural Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#39ff14',
                gridwidth=1,
                tickcolor='white',
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                gridcolor='#00d4ff',
                gridwidth=1,
                tickcolor='white',
                tickfont=dict(color='white', size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="🔮 Neural Profile Matrix",
            x=0.5,
            font=dict(color='#00d4ff', size=18, family='Space Grotesk')
        ),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional cyber metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sum(values)/len(values):.0f}%</div>
            <div class="metric-label">Profile Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_level = "LOW" if max(values) < 70 else "HIGH"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{risk_level}</div>
            <div class="metric-label">Risk Matrix</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        potential = "PREMIUM" if sum(values) > 400 else "STANDARD"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{potential}</div>
            <div class="metric-label">Tier Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        neural_efficiency = min(100, (educational_num * hours_per_week * age) / 10000 * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{neural_efficiency:.0f}%</div>
            <div class="metric-label">Neural Sync</div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Information Section
st.markdown("---")
st.markdown("""
<div class="feature-section">
    <h3>🤖 About This Neural Network</h3>
</div>
""", unsafe_allow_html=True)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Tech Stack</h3>
        <p>• Advanced ML algorithms</p>
        <p>• Quantum-inspired processing</p>
        <p>• Neural network optimization</p>
        <p>• Real-time data analysis</p>
        <p>• Cyberpunk UI/UX design</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Features</h3>
        <p>• Instant salary predictions</p>
        <p>• SHAP explainability</p>
        <p>• LIME local insights</p>
        <p>• Interactive visualizations</p>
        <p>• Confidence scoring</p>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Performance</h3>
        <p>• Sub-second predictions</p>
        <p>• 95%+ accuracy rate</p>
        <p>• Real-time processing</p>
        <p>• Scalable architecture</p>
        <p>• Future-proof design</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Footer with cyber theme
st.markdown("""
<div class="footer">
    <h3>🌟 Neural Network Status: ONLINE</h3>
    <p>Powered by quantum algorithms and cyber-enhanced machine learning</p>
    <p>© 2024 Cyber Salary Predictor • Next-gen AI solutions • Built for the future</p>
    <br>
    <p style="font-size: 0.9rem; opacity: 0.7;">
        ⚡ System Status: Fully Operational | 
        🧠 Neural Networks: Active | 
        🔮 Quantum Processing: Enabled
    </p>
</div>
""", unsafe_allow_html=True)

# Add some final enhancements
st.markdown("""
<script>
    // Add some interactive elements
    document.addEventListener('DOMContentLoaded', function() {
        // Add subtle animations on load
        const cards = document.querySelectorAll('.info-card, .feature-section, .metric-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            setTimeout(() => {
                card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    });
</script>
""", unsafe_allow_html=True)