"""
Sentimart Pro - Enterprise SaaS Analytics Dashboard
Professional ML Sentiment Analysis Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from urllib.parse import urlparse, parse_qs
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import io

# Optional imports with error handling
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False

try:
    from langdetect import detect
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sentimart Pro | Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENTERPRISE SAAS DASHBOARD CSS
# ============================================================================
st.markdown("""
<style>
    /* ===== DESIGN TOKENS ===== */
    :root {
        /* Colors */
        --primary: #4F46E5;
        --primary-hover: #4338CA;
        --success: #16A34A;
        --danger: #DC2626;
        --warning: #EA580C;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --bg-app: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-color: #E2E8F0;
        --border-light: #F1F5F9;
        
        /* Shadows */
        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        
        /* Spacing */
        --spacing-xs: 0.5rem;
        --spacing-sm: 0.75rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        
        /* Border Radius */
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
    }
    
    /* ===== GLOBAL OVERRIDES ===== */
    .main {
        padding: 2rem 3rem;
        background-color: var(--bg-app);
    }
    
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    p, span, div {
        color: var(--text-secondary);
    }
    
    /* ===== HEADER SECTION ===== */
    .dashboard-header {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        padding: 2rem 2.5rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
    }
    
    .dashboard-header h1 {
        color: white;
        font-size: 1.875rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
    }
    
    .dashboard-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 500;
        margin-top: 0.75rem;
        margin-right: 0.5rem;
    }
    
    /* ===== CARD SYSTEM ===== */
    .dashboard-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
        transition: box-shadow 0.2s ease;
    }
    
    .dashboard-card:hover {
        box-shadow: var(--shadow-md);
    }
    
    .card-header {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-light);
    }
    
    .card-subheader {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    
    /* ===== METRICS / KPI CARDS ===== */
    .kpi-card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-xs);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .kpi-card:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-sm);
    }
    
    .kpi-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-change {
        font-size: 0.75rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .kpi-change.positive {
        color: var(--success);
    }
    
    .kpi-change.negative {
        color: var(--danger);
    }
    
    /* ===== SENTIMENT RESULT CARDS ===== */
    .sentiment-result {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.75rem;
        border: 2px solid;
        box-shadow: var(--shadow-sm);
        margin: 1.5rem 0;
    }
    
    .sentiment-result.positive {
        border-color: var(--success);
        background: linear-gradient(to right, rgba(22, 163, 74, 0.03), rgba(22, 163, 74, 0.01));
    }
    
    .sentiment-result.negative {
        border-color: var(--danger);
        background: linear-gradient(to right, rgba(220, 38, 38, 0.03), rgba(220, 38, 38, 0.01));
    }
    
    .sentiment-result.neutral {
        border-color: var(--primary);
        background: linear-gradient(to right, rgba(79, 70, 229, 0.03), rgba(79, 70, 229, 0.01));
    }
    
    .sentiment-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .sentiment-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .sentiment-label {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .sentiment-meta {
        display: flex;
        gap: 1.5rem;
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid var(--border-light);
    }
    
    .sentiment-meta-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sentiment-meta-label {
        font-weight: 500;
        color: var(--text-muted);
    }
    
    .sentiment-meta-value {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border-color);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] .element-container {
        padding: 0 1rem;
    }
    
    .sidebar-section {
        background: var(--bg-app);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-light);
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    
    .status-row {
        display: flex;
        align-items: center;
        padding: 0.625rem 0;
        font-size: 0.875rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.625rem;
        flex-shrink: 0;
    }
    
    .status-dot.active {
        background: var(--success);
        box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.2);
    }
    
    .status-dot.inactive {
        background: var(--text-muted);
    }
    
    .status-label {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-xs);
    }
    
    .stButton > button:hover {
        background-color: var(--primary-hover);
        box-shadow: var(--shadow-sm);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button[kind="primary"] {
        background-color: var(--primary);
    }
    
    .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: var(--primary);
        border: 1px solid var(--primary);
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        border-radius: 0;
        padding: 0.875rem 1.25rem;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.875rem;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background-color: var(--bg-app);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom-color: var(--primary);
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    
    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.625rem 0.875rem;
        font-size: 0.875rem;
        color: var(--text-primary);
        background-color: var(--bg-card);
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        outline: none;
    }
    
    .stTextArea > div > div > textarea {
        min-height: 120px;
    }
    
    /* ===== CHECKBOX & RADIO ===== */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    .stCheckbox label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card);
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background-color: rgba(79, 70, 229, 0.02);
    }
    
    [data-testid="stFileUploader"] section {
        border: none;
        padding: 0;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: var(--bg-app);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.875rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--bg-card);
        border-color: var(--primary);
    }
    
    /* ===== CHART CONTAINER ===== */
    .chart-wrapper {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-xs);
        margin: 1rem 0;
    }
    
    .chart-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ===== DATA TABLE ===== */
    .dataframe-container {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-xs);
        margin: 1.5rem 0;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background-color: var(--bg-card);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-xs);
    }
    
    .stAlert[data-baseweb="notification"] {
        background-color: var(--bg-card);
    }
    
    /* Info alert */
    div[data-baseweb="notification"][kind="info"] {
        background-color: rgba(79, 70, 229, 0.05);
        border-color: var(--primary);
    }
    
    /* Success alert */
    div[data-baseweb="notification"][kind="success"] {
        background-color: rgba(22, 163, 74, 0.05);
        border-color: var(--success);
    }
    
    /* Warning alert */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: rgba(234, 88, 12, 0.05);
        border-color: var(--warning);
    }
    
    /* Error alert */
    div[data-baseweb="notification"][kind="error"] {
        background-color: rgba(220, 38, 38, 0.05);
        border-color: var(--danger);
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background-color: var(--primary);
        border-radius: var(--radius-sm);
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background-color: var(--success);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        font-size: 0.8125rem;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        background-color: #15803D;
        box-shadow: var(--shadow-sm);
    }
    
    /* ===== ACTIVITY TIMELINE ===== */
    .activity-item {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        border-left: 3px solid var(--primary);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .activity-item:hover {
        box-shadow: var(--shadow-sm);
        border-left-width: 3px;
    }
    
    .activity-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    
    .activity-sentiment {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .activity-sentiment-icon {
        font-size: 1.25rem;
    }
    
    .activity-sentiment-label {
        font-weight: 600;
        font-size: 0.9375rem;
    }
    
    .activity-confidence {
        font-size: 0.8125rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    .activity-text {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    
    .activity-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background: rgba(79, 70, 229, 0.05);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: var(--text-primary);
    }
    
    .success-box {
        background: rgba(22, 163, 74, 0.05);
        border: 1px solid rgba(22, 163, 74, 0.2);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: var(--text-primary);
    }
    
    .warning-box {
        background: rgba(234, 88, 12, 0.05);
        border: 1px solid rgba(234, 88, 12, 0.2);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: var(--text-primary);
    }
    
    /* ===== BADGE ===== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    
    .badge-primary {
        background-color: rgba(79, 70, 229, 0.1);
        color: var(--primary);
    }
    
    .badge-success {
        background-color: rgba(22, 163, 74, 0.1);
        color: var(--success);
    }
    
    .badge-danger {
        background-color: rgba(220, 38, 38, 0.1);
        color: var(--danger);
    }
    
    /* ===== SECTION DIVIDER ===== */
    .section-divider {
        height: 1px;
        background: var(--border-color);
        margin: 2rem 0;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .dashboard-header {
            padding: 1.5rem;
        }
        
        .dashboard-header h1 {
            font-size: 1.5rem;
        }
        
        .dashboard-card {
            padding: 1rem;
        }
        
        .kpi-value {
            font-size: 1.75rem;
        }
    }
    
    /* ===== UTILITY CLASSES ===== */
    .text-primary { color: var(--text-primary); }
    .text-secondary { color: var(--text-secondary); }
    .text-muted { color: var(--text-muted); }
    .text-success { color: var(--success); }
    .text-danger { color: var(--danger); }
    
    .font-semibold { font-weight: 600; }
    .font-bold { font-weight: 700; }
    
    .mb-1 { margin-bottom: 0.5rem; }
    .mb-2 { margin-bottom: 1rem; }
    .mb-3 { margin-bottom: 1.5rem; }
    
    .mt-1 { margin-top: 0.5rem; }
    .mt-2 { margin-top: 1rem; }
    .mt-3 { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE (NO CHANGES)
# ============================================================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0

# ============================================================================
# MODEL LOADING (NO CHANGES)
# ============================================================================
@st.cache_resource
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("model")
        model = AutoModelForSequenceClassification.from_pretrained("model")
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

@st.cache_resource
def load_emotion_model():
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=-1
        )
        return emotion_classifier, True
    except Exception as e:
        return None, False

@st.cache_resource
def load_multilingual_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        return tokenizer, model, True
    except Exception as e:
        return None, None, False

# ============================================================================
# PREDICTION FUNCTIONS (NO CHANGES)
# ============================================================================
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        positive_prob = probs[0][1].item() if probs.shape[1] > 1 else 0
        negative_prob = probs[0][0].item()
    
    return {
        'label': "Positive" if pred_class == 1 else "Negative",
        'confidence': confidence,
        'positive_prob': positive_prob,
        'negative_prob': negative_prob
    }

def predict_emotions(text, emotion_classifier):
    try:
        results = emotion_classifier(text[:512])[0]
        emotions = {item['label']: item['score'] for item in results}
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        return {'emotions': emotions, 'top_emotion': top_emotion[0], 'top_score': top_emotion[1]}
    except:
        return None

def predict_multilingual(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            rating = torch.argmax(probs, dim=1).item() + 1
        label = "Positive" if rating >= 4 else "Negative" if rating <= 2 else "Neutral"
        return {'rating': rating, 'label': label, 'confidence': probs[0][rating-1].item()}
    except:
        return None

def detect_language(text):
    if not LANG_DETECT_AVAILABLE:
        return 'en'
    try:
        return detect(text)
    except:
        return 'en'

# ============================================================================
# API FUNCTIONS (NO CHANGES)
# ============================================================================
def extract_youtube_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
    return None

def fetch_youtube_comments(video_url, api_key, max_results=50):
    if not YOUTUBE_API_AVAILABLE or not api_key:
        return [
            {"text": "This video is amazing! Really helpful content.", "author": "User1"},
            {"text": "Thanks for sharing. Learned a lot!", "author": "User2"},
            {"text": "Not sure about this. Needs more explanation.", "author": "User3"},
            {"text": "Excellent tutorial! Subscribed!", "author": "User4"},
            {"text": "Didn't work for me. Got errors.", "author": "User5"},
        ][:max_results], "demo"
    
    try:
        video_id = extract_youtube_video_id(video_url)
        if not video_id:
            return None, "Invalid URL"
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText"
        )
        response = request.execute()
        
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text': comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'likes': comment['likeCount']
            })
        return comments, "api"
    except Exception as e:
        return None, str(e)

# ============================================================================
# FILE READING (NO CHANGES)
# ============================================================================
def detect_encoding(file_bytes):
    if CHARDET_AVAILABLE:
        try:
            result = chardet.detect(file_bytes)
            return result['encoding'] if result['encoding'] else 'utf-8'
        except:
            return 'utf-8'
    return 'utf-8'

def read_pdf(file):
    if not PDF_AVAILABLE:
        return None, "PyPDF2 not installed. Install with: pip install PyPDF2"
    try:
        pdf = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if not text.strip():
            return None, "Could not extract text from PDF"
        
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
        return paragraphs, None
    except Exception as e:
        return None, f"PDF read error: {str(e)}"

def read_csv_with_fallback(uploaded_file):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']
    
    if CHARDET_AVAILABLE:
        try:
            uploaded_file.seek(0)
            raw_data = uploaded_file.read()
            detected_encoding = detect_encoding(raw_data)
            if detected_encoding and detected_encoding.lower() not in [e.lower() for e in encodings]:
                encodings.insert(0, detected_encoding)
        except:
            pass
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file, 
                encoding=encoding,
                on_bad_lines='skip',
                engine='python'
            )
            return df, None, encoding
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception as e:
            continue
    
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            encoding='utf-8',
            errors='ignore',
            on_bad_lines='skip',
            engine='python'
        )
        return df, None, 'utf-8 (with errors ignored)'
    except Exception as e:
        return None, f"Could not read CSV: {str(e)}", None

def read_excel_with_fallback(uploaded_file):
    try:
        uploaded_file.seek(0)
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df, None
        except:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            return df, None
    except Exception as e:
        return None, f"Excel read error: {str(e)}. Make sure openpyxl is installed: pip install openpyxl"

def read_text_with_fallback(uploaded_file):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode(encoding)
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            return lines, None, encoding
        except UnicodeDecodeError:
            continue
    
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        return lines, None, 'utf-8 (with errors ignored)'
    except Exception as e:
        return None, f"Could not read text file: {str(e)}", None

def read_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if ext == 'csv':
            df, error, encoding = read_csv_with_fallback(uploaded_file)
            if error:
                return None, 'error', error, None
            return df, 'dataframe', None, encoding
        
        elif ext in ['xlsx', 'xls']:
            df, error = read_excel_with_fallback(uploaded_file)
            if error:
                return None, 'error', error, None
            return df, 'dataframe', None, 'excel'
        
        elif ext == 'txt':
            lines, error, encoding = read_text_with_fallback(uploaded_file)
            if error:
                return None, 'error', error, None
            return lines, 'text', None, encoding
        
        elif ext == 'pdf':
            paragraphs, error = read_pdf(uploaded_file)
            if error:
                return None, 'error', error, None
            return paragraphs, 'text', None, 'pdf'
        
        else:
            return None, 'unsupported', f"Unsupported file format: .{ext}", None
    
    except Exception as e:
        return None, 'error', f"Unexpected error reading file: {str(e)}", None

# ============================================================================
# VISUALIZATIONS (NO CHANGES)
# ============================================================================
def create_confidence_chart(pos, neg):
    fig = go.Figure(data=[
        go.Bar(
            x=['Negative', 'Positive'],
            y=[neg, pos],
            marker_color=['#DC2626', '#16A34A'],
            text=[f'{neg:.1%}', f'{pos:.1%}'],
            textposition='auto',
            textfont=dict(size=14, weight=600)
        )
    ])
    fig.update_layout(
        title=None,
        height=280,
        showlegend=False,
        yaxis=dict(tickformat='.0%', gridcolor='#F1F5F9'),
        xaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family='system-ui', size=12, color='#475569')
    )
    return fig

def create_emotion_chart(emotions):
    names = list(emotions.keys())
    scores = list(emotions.values())
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=scores,
            marker_color='#4F46E5',
            text=[f'{s:.1%}' for s in scores],
            textposition='auto',
            textfont=dict(size=12, weight=600)
        )
    ])
    fig.update_layout(
        title=None,
        height=280,
        yaxis=dict(tickformat='.0%', gridcolor='#F1F5F9'),
        xaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family='system-ui', size=12, color='#475569')
    )
    return fig

def create_wordcloud(texts, sentiment=None):
    text = " ".join(texts)
    if not text.strip():
        return None
    colormap = 'Greens' if sentiment == 'positive' else 'Reds' if sentiment == 'negative' else 'viridis'
    wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_distribution_pie(df):
    counts = df['sentiment'].value_counts()
    colors = ['#16A34A' if l == 'Positive' else '#DC2626' for l in counts.index]
    fig = go.Figure(data=[
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker_colors=colors,
            textfont=dict(size=14, weight=600)
        )
    ])
    fig.update_layout(
        title=None,
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family='system-ui', size=12, color='#475569')
    )
    return fig

# ============================================================================
# MAIN APP - ENTERPRISE DASHBOARD LAYOUT
# ============================================================================
def main():
    # DASHBOARD HEADER
    st.markdown("""
    <div class="dashboard-header">
        <h1>📊 Sentimart</h1>
        <p>Enterprise Sentiment & Emotion Analysis Platform</p>
        <div style="margin-top: 1rem;">
            <span class="header-badge">YouTube API</span>
            <span class="header-badge">PDF Support</span>
            <span class="header-badge">Batch Processing</span>
            <span class="header-badge">Multi-Language</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    tokenizer, model, loaded = load_sentiment_model()
    emotion_clf, emotion_ok = load_emotion_model()
    ml_tok, ml_model, ml_ok = load_multilingual_model()
    
    if not loaded:
        st.error("⚠️ Model failed to load. Please check your model directory.")
        st.stop()
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)
        
        with st.expander("API Keys", expanded=False):
            youtube_key = st.text_input("YouTube API Key", type="password", help="Optional for real data")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">System Status</div>', unsafe_allow_html=True)
        
        status_html = f"""
        <div class="status-row">
            <div class="status-dot {'active' if loaded else 'inactive'}"></div>
            <span class="status-label">Sentiment Model</span>
        </div>
        <div class="status-row">
            <div class="status-dot {'active' if emotion_ok else 'inactive'}"></div>
            <span class="status-label">Emotion Detection</span>
        </div>
        <div class="status-row">
            <div class="status-dot {'active' if ml_ok else 'inactive'}"></div>
            <span class="status-label">Multi-Language</span>
        </div>
        <div class="status-row">
            <div class="status-dot {'active' if PDF_AVAILABLE else 'inactive'}"></div>
            <span class="status-label">PDF Support</span>
        </div>
        <div class="status-row">
            <div class="status-dot {'active' if CHARDET_AVAILABLE else 'inactive'}"></div>
            <span class="status-label">Encoding Detection</span>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Analytics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{st.session_state.total_analyses}</div>
            <div class="kpi-label">Total Analyses</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.batch_results is not None:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            csv = st.session_state.batch_results.to_csv(index=False)
            st.download_button(
                "Export Results",
                csv,
                f"sentimart_export_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TABS ==========
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Analysis",
        "YouTube Comments",
        "Batch Upload",
        "Analytics Dashboard"
    ])
    
    # ========== TAB 1: SINGLE ANALYSIS ==========
    with tab1:
        with st.container():
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Text Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subheader">Analyze sentiment, emotions, and language from any text input</div>', unsafe_allow_html=True)
            
            text = st.text_area(
                "Input Text",
                height=140,
                placeholder="Enter your text here for sentiment analysis...",
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                do_emotion = st.checkbox("Include Emotion Analysis", value=emotion_ok, disabled=not emotion_ok)
            with col2:
                do_ml = st.checkbox("Multi-Language Support", value=ml_ok, disabled=not ml_ok)
            with col3:
                show_wc = st.checkbox("Generate Word Cloud", value=False)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Run Analysis", type="primary", use_container_width=True):
                if not text.strip():
                    st.warning("Please enter text to analyze")
                else:
                    with st.spinner("Analyzing..."):
                        result = predict_sentiment(text, tokenizer, model)
                        lang = detect_language(text)
                        
                        emotion_res = predict_emotions(text, emotion_clf) if do_emotion and emotion_ok else None
                        ml_res = predict_multilingual(text, ml_tok, ml_model) if do_ml and ml_ok and lang != 'en' else None
                        
                        st.session_state.total_analyses += 1
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'text': text[:80],
                            'sentiment': result['label'],
                            'confidence': result['confidence'],
                            'language': lang
                        })
                    
                    # Result Card
                    sentiment_class = result['label'].lower()
                    emoji = "✓" if result['label'] == "Positive" else "✗"
                    
                    st.markdown(f"""
                    <div class="sentiment-result {sentiment_class}">
                        <div class="sentiment-header">
                            <div style="display: flex; align-items: center;">
                                <div class="sentiment-icon">{emoji}</div>
                                <div>
                                    <div class="sentiment-label">{result['label']} Sentiment</div>
                                </div>
                            </div>
                        </div>
                        <div class="sentiment-meta">
                            <div class="sentiment-meta-item">
                                <span class="sentiment-meta-label">Confidence:</span>
                                <span class="sentiment-meta-value">{result['confidence']:.1%}</span>
                            </div>
                            <div class="sentiment-meta-item">
                                <span class="sentiment-meta-label">Language:</span>
                                <span class="sentiment-meta-value">{lang.upper()}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">Confidence Distribution</div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            create_confidence_chart(result['positive_prob'], result['negative_prob']),
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if emotion_res:
                        with col2:
                            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                            st.markdown('<div class="chart-title">Emotion Analysis</div>', unsafe_allow_html=True)
                            st.plotly_chart(
                                create_emotion_chart(emotion_res['emotions']),
                                use_container_width=True,
                                config={'displayModeBar': False}
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    if ml_res:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Multi-Language Analysis:</strong> {ml_res['label']} sentiment detected 
                            (Rating: {ml_res['rating']}/5, Confidence: {ml_res['confidence']:.1%})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if show_wc:
                        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">Word Cloud</div>', unsafe_allow_html=True)
                        wc = create_wordcloud([text])
                        if wc:
                            st.pyplot(wc)
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TAB 2: YOUTUBE ==========
    with tab2:
        with st.container():
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">YouTube Comments Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subheader">Extract and analyze sentiment from YouTube video comments</div>', unsafe_allow_html=True)
            
            url = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                max_items = st.slider("Maximum Comments", 10, 100, 30)
            with col2:
                show_wc_url = st.checkbox("Include Word Cloud", value=True, key="wc_url")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Fetch & Analyze Comments", type="primary", use_container_width=True):
                if not url:
                    st.warning("Please enter a YouTube URL")
                elif 'youtube' not in url and 'youtu.be' not in url:
                    st.error("Invalid YouTube URL")
                else:
                    with st.spinner("Fetching comments..."):
                        comments, mode = fetch_youtube_comments(url, youtube_key if youtube_key else None, max_items)
                        
                        if isinstance(mode, str) and mode in ['demo', 'api']:
                            if mode == 'demo':
                                st.markdown("""
                                <div class="info-box">
                                    <strong>Demo Mode:</strong> Displaying sample comments. Add API key for real data.
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="success-box">
                                    <strong>Connected:</strong> Fetching real YouTube comments via API.
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error(f"Error: {mode}")
                            st.stop()
                        
                        if comments:
                            results = []
                            prog = st.progress(0)
                            for i, c in enumerate(comments):
                                if c['text']:
                                    r = predict_sentiment(c['text'], tokenizer, model)
                                    results.append({
                                        'text': c['text'][:70],
                                        'sentiment': r['label'],
                                        'confidence': r['confidence'],
                                        'author': c.get('author', 'Unknown')
                                    })
                                prog.progress((i+1)/len(comments))
                            
                            if results:
                                df = pd.DataFrame(results)
                                st.session_state.batch_results = df
                                st.session_state.total_analyses += len(results)
                                
                                # KPI Metrics
                                col1, col2, col3 = st.columns(3)
                                pos = len(df[df['sentiment']=='Positive'])
                                neg = len(df) - pos
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="kpi-card">
                                        <div class="kpi-value">{len(df)}</div>
                                        <div class="kpi-label">Total Comments</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="kpi-card">
                                        <div class="kpi-value text-success">{pos}</div>
                                        <div class="kpi-label">Positive</div>
                                        <div class="kpi-change positive">+{pos/len(df)*100:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="kpi-card">
                                        <div class="kpi-value text-danger">{neg}</div>
                                        <div class="kpi-label">Negative</div>
                                        <div class="kpi-change negative">{neg/len(df)*100:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Charts
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                                    st.markdown('<div class="chart-title">Sentiment Distribution</div>', unsafe_allow_html=True)
                                    st.plotly_chart(
                                        create_distribution_pie(df),
                                        use_container_width=True,
                                        config={'displayModeBar': False}
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                if show_wc_url:
                                    with col2:
                                        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                                        st.markdown('<div class="chart-title">Word Cloud</div>', unsafe_allow_html=True)
                                        wc = create_wordcloud(df['text'].tolist())
                                        if wc:
                                            st.pyplot(wc)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Data Table
                                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                                st.dataframe(df, use_container_width=True, height=400)
                                st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TAB 3: BATCH UPLOAD ==========
    with tab3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Batch File Processing</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subheader">Upload CSV, Excel, TXT, or PDF files for bulk sentiment analysis</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Supported formats:</strong> CSV, Excel (.xlsx, .xls), TXT, PDF
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        file = st.file_uploader(
            "Upload File",
            type=['csv', 'xlsx', 'xls', 'txt', 'pdf'],
            label_visibility="collapsed"
        )
        
        if file:
            with st.spinner("Processing file..."):
                content, ftype, error, encoding_info = read_file(file)
            
            if error:
                st.error(f"Error: {error}")
                if 'encoding' in error.lower():
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Encoding issue detected.</strong> Try saving the file as UTF-8 or install chardet: <code>pip install chardet</code>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif ftype == 'dataframe':
                st.markdown(f"""
                <div class="success-box">
                    Successfully loaded <strong>{len(content)} rows</strong>
                    {f' • Encoding: {encoding_info}' if encoding_info else ''}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Preview Data", expanded=True):
                    st.dataframe(content.head(10), use_container_width=True)
                
                # Find text column
                col = None
                possible_cols = ['review', 'text', 'comment', 'feedback', 'message', 'content']
                
                for possible in possible_cols:
                    for c in content.columns:
                        if possible in c.lower():
                            col = c
                            break
                    if col:
                        break
                
                if col:
                    st.markdown(f"""
                    <div class="info-box">
                        Text column detected: <strong>{col}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    max_rows = st.slider("Maximum Rows to Process", 10, 500, 100)
                    
                    if st.button("Start Batch Analysis", type="primary", use_container_width=True):
                        valid = content[content[col].notna()][col]
                        results = []
                        prog = st.progress(0)
                        
                        for i, txt in enumerate(valid.head(max_rows)):
                            try:
                                txt_str = str(txt)
                                if txt_str.strip():
                                    r = predict_sentiment(txt_str, tokenizer, model)
                                    results.append({
                                        'review': txt_str[:80],
                                        'sentiment': r['label'],
                                        'confidence': r['confidence']
                                    })
                            except Exception as e:
                                continue
                            prog.progress((i+1)/min(len(valid), max_rows))
                        
                        if results:
                            df = pd.DataFrame(results)
                            st.session_state.batch_results = df
                            st.session_state.total_analyses += len(results)
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            pos = len(df[df['sentiment']=='Positive'])
                            
                            with col1:
                                st.markdown(f"""
                                <div class="kpi-card">
                                    <div class="kpi-value">{len(results)}</div>
                                    <div class="kpi-label">Processed</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="kpi-card">
                                    <div class="kpi-value text-success">{pos}</div>
                                    <div class="kpi-label">Positive</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="kpi-card">
                                    <div class="kpi-value text-danger">{len(results)-pos}</div>
                                    <div class="kpi-label">Negative</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                            st.plotly_chart(
                                create_distribution_pie(df),
                                use_container_width=True,
                                config={'displayModeBar': False}
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(df, use_container_width=True, height=400)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No valid text found")
                else:
                    st.error("No text column detected")
                    st.info(f"Available columns: {', '.join(content.columns)}")
                    selected_col = st.selectbox("Select text column:", content.columns)
                    if st.button("Use Selected Column"):
                        col = selected_col
                        st.rerun()
            
            elif ftype == 'text':
                st.markdown(f"""
                <div class="success-box">
                    Found <strong>{len(content)} text items</strong>
                    {f' • Encoding: {encoding_info}' if encoding_info else ''}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Preview", expanded=True):
                    for i, item in enumerate(content[:5]):
                        st.text(f"{i+1}. {item[:100]}...")
                
                max_items = st.slider("Maximum Items", 10, 500, min(100, len(content)))
                
                if st.button("Start Batch Analysis", type="primary", use_container_width=True):
                    results = []
                    prog = st.progress(0)
                    
                    for i, txt in enumerate(content[:max_items]):
                        try:
                            if txt.strip():
                                r = predict_sentiment(txt, tokenizer, model)
                                results.append({
                                    'text': txt[:80],
                                    'sentiment': r['label'],
                                    'confidence': r['confidence']
                                })
                        except:
                            continue
                        prog.progress((i+1)/min(len(content), max_items))
                    
                    if results:
                        df = pd.DataFrame(results)
                        st.session_state.batch_results = df
                        st.session_state.total_analyses += len(results)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        pos = len(df[df['sentiment']=='Positive'])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="kpi-card">
                                <div class="kpi-value">{len(results)}</div>
                                <div class="kpi-label">Processed</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="kpi-card">
                                <div class="kpi-value text-success">{pos}</div>
                                <div class="kpi-label">Positive</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="kpi-card">
                                <div class="kpi-value text-danger">{len(results)-pos}</div>
                                <div class="kpi-label">Negative</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
                        st.plotly_chart(
                            create_distribution_pie(df),
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True, height=400)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No valid text found")
    
    # ========== TAB 4: ANALYTICS ==========
    with tab4:
        st.markdown('<div class="card-header">Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subheader">Executive overview of all sentiment analysis activity</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_history:
            df = pd.DataFrame(st.session_state.analysis_history)
            
            # Top KPIs
            total = len(df)
            pos = len(df[df['sentiment']=='Positive'])
            avg = df['confidence'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{total}</div>
                    <div class="kpi-label">Total Analyses</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value text-success">{pos/total:.1%}</div>
                    <div class="kpi-label">Positive Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{avg:.1%}</div>
                    <div class="kpi-label">Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Recent Activity</div>', unsafe_allow_html=True)
            
            for _, row in df.tail(10).sort_values('timestamp', ascending=False).iterrows():
                emoji = "✓" if row['sentiment'] == "Positive" else "✗"
                sentiment_class = "text-success" if row['sentiment'] == "Positive" else "text-danger"
                
                st.markdown(f"""
                <div class="activity-item">
                    <div class="activity-header">
                        <div class="activity-sentiment">
                            <span class="activity-sentiment-icon">{emoji}</span>
                            <span class="activity-sentiment-label {sentiment_class}">{row['sentiment']}</span>
                        </div>
                        <span class="activity-confidence">{row['confidence']:.1%}</span>
                    </div>
                    <div class="activity-text">{row['text']}</div>
                    <div class="activity-meta">
                        <span>🌍 {row.get('language', 'en').upper()}</span>
                        <span>📅 {row['timestamp']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dashboard-card" style="text-align: center; padding: 4rem 2rem;">
                <h3 style="color: var(--text-muted);">No Activity Yet</h3>
                <p style="color: var(--text-muted); margin-top: 1rem;">
                    Start analyzing text to populate your dashboard with insights
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()