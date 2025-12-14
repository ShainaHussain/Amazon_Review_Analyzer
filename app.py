"""
Sentimart Pro - Advanced Multi-Source Sentiment Analysis
Enhanced UI/UX Version - Production-Ready Design
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
# PAGE CONFIG - Enhanced for Premium Feel
# ============================================================================
st.set_page_config(
    page_title="Sentimart Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PREMIUM CSS STYLING - Production-Grade Design System
# ============================================================================
st.markdown("""
<style>
    /* ===== DESIGN SYSTEM ===== */
    :root {
        --primary: #667eea;
        --primary-dark: #5568d3;
        --secondary: #764ba2;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-700: #374151;
        --neutral-800: #1f2937;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== GLOBAL OVERRIDES ===== */
    .main {
        padding: 1rem 2rem 3rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f2f5 100%);
    }
    
    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ===== HERO HEADER ===== */
    .hero-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem 3rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-features {
        font-size: 0.9rem;
        opacity: 0.85;
        position: relative;
        z-index: 1;
    }
    
    /* ===== PREMIUM CARDS ===== */
    .premium-card {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--neutral-200);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    /* ===== SENTIMENT RESULT CARDS ===== */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-lg);
        border-left: 5px solid;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: var(--shadow-xl);
        transform: translateY(-2px);
    }
    
    .result-positive {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: var(--success);
        color: #065f46;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left-color: var(--danger);
        color: #991b1b;
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        border-left-color: var(--primary);
        color: #3730a3;
    }
    
    .result-card h3 {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .result-meta {
        font-size: 0.95rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--neutral-200);
        box-shadow: var(--shadow-sm);
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-md);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--neutral-700);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-right: 1px solid var(--neutral-200);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        padding: 0.5rem 0;
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border: 1px solid var(--neutral-200);
        box-shadow: var(--shadow-sm);
    }
    
    .sidebar-header {
        font-size: 0.875rem;
        font-weight: 700;
        color: var(--neutral-700);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--neutral-200);
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-active {
        background: var(--success);
        box-shadow: 0 0 8px var(--success);
    }
    
    .status-inactive {
        background: var(--neutral-200);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%);
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: var(--shadow-sm);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--neutral-100);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white !important;
    }
    
    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid var(--neutral-200);
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed var(--neutral-200);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background: var(--neutral-50);
    }
    
    /* ===== CHARTS CONTAINER ===== */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
        border: 1px solid var(--neutral-200);
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .dataframe-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-md);
        margin: 1.5rem 0;
        border: 1px solid var(--neutral-200);
    }
    
    /* ===== ALERTS & NOTIFICATIONS ===== */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        border: 1px solid var(--neutral-200);
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--neutral-50);
        border-color: var(--primary);
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 10px;
    }
    
    /* ===== EMOTION BADGE ===== */
    .emotion-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        color: var(--primary);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .emotion-badge:hover {
        transform: scale(1.05);
        box-shadow: var(--shadow-md);
    }
    
    /* ===== ACTIVITY CARD ===== */
    .activity-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border-left: 4px solid var(--primary);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .activity-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateX(4px);
    }
    
    .activity-text {
        font-size: 0.95rem;
        color: var(--neutral-800);
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .activity-meta {
        font-size: 0.8rem;
        color: var(--neutral-700);
        opacity: 0.8;
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--success);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--warning);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .hero-header h1 {
            font-size: 1.75rem;
        }
        
        .hero-subtitle {
            font-size: 0.95rem;
        }
        
        .premium-card {
            padding: 1.25rem;
        }
        
        .result-card {
            padding: 1.5rem;
        }
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--neutral-800);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid var(--primary);
        display: inline-block;
    }
    
    /* ===== DOWNLOAD BUTTON SPECIAL ===== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
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
        go.Bar(x=['Negative', 'Positive'], y=[neg, pos],
               marker_color=['#ef4444', '#10b981'],
               text=[f'{neg:.1%}', f'{pos:.1%}'], textposition='auto')
    ])
    fig.update_layout(
        title="Confidence Distribution",
        height=300,
        showlegend=False,
        yaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_emotion_chart(emotions):
    names = list(emotions.keys())
    scores = list(emotions.values())
    fig = go.Figure(data=[
        go.Bar(x=names, y=scores, marker_color='#667eea',
               text=[f'{s:.1%}' for s in scores], textposition='auto')
    ])
    fig.update_layout(
        title="Emotion Analysis",
        height=300,
        yaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
    colors = ['#10b981' if l == 'Positive' else '#ef4444' for l in counts.index]
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.4, marker_colors=colors)])
    fig.update_layout(
        title="Sentiment Distribution",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ============================================================================
# MAIN APP - ENHANCED UI/UX
# ============================================================================
def main():
    # HERO HEADER
    st.markdown("""
    <div class="hero-header">
        <h1>🛒 Sentimart Pro</h1>
        <p class="hero-subtitle">Advanced Multi-Source Sentiment & Emotion Analysis</p>
        <p class="hero-features">✨ YouTube API • 📄 PDF Support • ☁️ Word Clouds • 😊 Emotion Detection • 🌍 Multi-Language</p>
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
        st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        with st.expander("🔑 API Keys", expanded=False):
            youtube_key = st.text_input("YouTube API Key", type="password", help="Optional - enables real YouTube data")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">🤖 System Status</div>', unsafe_allow_html=True)
        
        status_html = f"""
        <div style="padding: 0.5rem 0;">
            <span class="status-indicator {'status-active' if loaded else 'status-inactive'}"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Sentiment Model</span>
        </div>
        <div style="padding: 0.5rem 0;">
            <span class="status-indicator {'status-active' if emotion_ok else 'status-inactive'}"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Emotion Detection</span>
        </div>
        <div style="padding: 0.5rem 0;">
            <span class="status-indicator {'status-active' if ml_ok else 'status-inactive'}"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Multi-Language</span>
        </div>
        <div style="padding: 0.5rem 0;">
            <span class="status-indicator {'status-active' if PDF_AVAILABLE else 'status-inactive'}"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">PDF Support</span>
        </div>
        <div style="padding: 0.5rem 0;">
            <span class="status-indicator {'status-active' if CHARDET_AVAILABLE else 'status-inactive'}"></span>
            <span style="font-size: 0.9rem; font-weight: 500;">Advanced Encoding</span>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        
        if not CHARDET_AVAILABLE:
            st.caption("💡 Install chardet: `pip install chardet`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">📊 Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_analyses}</div>
            <div class="metric-label">Total Analyses</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.batch_results is not None:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            csv = st.session_state.batch_results.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                f"sentimart_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TABS ==========
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Analysis", "🔗 YouTube", "📦 Batch Upload", "📊 Analytics"])
    
    # ========== TAB 1: SINGLE ANALYSIS ==========
    with tab1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Analyze Single Text")
        st.caption("Get instant sentiment, emotion, and multi-language insights")
        
        text = st.text_area(
            "Enter your text:",
            height=150,
            placeholder="Paste your review, comment, or feedback here...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            do_emotion = st.checkbox("🎭 Emotion Analysis", value=emotion_ok, disabled=not emotion_ok)
        with col2:
            do_ml = st.checkbox("🌍 Multi-Language", value=ml_ok, disabled=not ml_ok)
        with col3:
            show_wc = st.checkbox("☁️ Word Cloud", value=False)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Now", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("⚠️ Please enter some text to analyze")
            else:
                with st.spinner("🧠 Analyzing your text..."):
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
                cls = "result-positive" if result['label'] == "Positive" else "result-negative"
                emoji = "😊" if result['label'] == "Positive" else "😠"
                
                st.markdown(f"""
                <div class="result-card {cls}">
                    <h3>{emoji} {result['label']} Sentiment</h3>
                    <p class="result-meta">
                        <strong>Confidence:</strong> {result['confidence']:.1%} | 
                        <strong>Language:</strong> {lang.upper()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(create_confidence_chart(result['positive_prob'], result['negative_prob']), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if emotion_res:
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(create_emotion_chart(emotion_res['emotions']), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                if ml_res:
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>🌍 Multi-Language Analysis:</strong> {ml_res['label']} 
                        (Rating: {ml_res['rating']}/5, Confidence: {ml_res['confidence']:.1%})
                    </div>
                    """, unsafe_allow_html=True)
                
                if show_wc:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    wc = create_wordcloud([text])
                    if wc:
                        st.pyplot(wc)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TAB 2: YOUTUBE ==========
    with tab2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 🔗 YouTube Comments Analysis")
        st.caption("Analyze sentiment from YouTube video comments")
        
        url = st.text_input(
            "YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_items = st.slider("Maximum comments to analyze:", 10, 100, 30)
        with col2:
            show_wc_url = st.checkbox("☁️ Generate Word Cloud", value=True, key="wc_url")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True):
            if not url:
                st.warning("⚠️ Please enter a YouTube URL")
            elif 'youtube' not in url and 'youtu.be' not in url:
                st.error("❌ Please enter a valid YouTube URL")
            else:
                with st.spinner("🔍 Fetching YouTube comments..."):
                    comments, mode = fetch_youtube_comments(url, youtube_key if youtube_key else None, max_items)
                    
                    if isinstance(mode, str) and mode in ['demo', 'api']:
                        if mode == 'demo':
                            st.markdown("""
                            <div class="info-box">
                                🔵 <strong>Demo Mode Active:</strong> Showing sample comments. 
                                Add your YouTube API key in the sidebar for real data.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                                ✅ <strong>Connected to YouTube API</strong> - Fetching real comments
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Error: {mode}")
                        st.stop()
                    
                    if comments:
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ Successfully fetched <strong>{len(comments)} comments</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
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
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            pos = len(df[df['sentiment']=='Positive'])
                            neg = len(df) - pos
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{len(df)}</div>
                                    <div class="metric-label">Total Comments</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #10b981;">{pos}</div>
                                    <div class="metric-label">Positive</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #ef4444;">{neg}</div>
                                    <div class="metric-label">Negative</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Charts
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                st.plotly_chart(create_distribution_pie(df), use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if show_wc_url:
                                with col2:
                                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                    wc = create_wordcloud(df['text'].tolist())
                                    if wc:
                                        st.pyplot(wc)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Data table
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(df, use_container_width=True, height=400)
                            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== TAB 3: FILE UPLOAD ==========
    with tab3:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 📦 Batch File Analysis")
        st.caption("Upload CSV, Excel, TXT, or PDF files for bulk sentiment analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="info-box">
                ✅ <strong>Supported Formats:</strong> CSV, Excel (.xlsx, .xls), TXT, PDF
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if not CHARDET_AVAILABLE:
                st.markdown("""
                <div class="warning-box">
                    💡 Install chardet for better encoding detection
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        file = st.file_uploader("Upload your file:", type=['csv', 'xlsx', 'xls', 'txt', 'pdf'], label_visibility="collapsed")
        
        if file:
            with st.spinner("📖 Reading file..."):
                content, ftype, error, encoding_info = read_file(file)
            
            if error:
                st.error(f"❌ {error}")
                if 'encoding' in error.lower() or 'utf' in error.lower():
                    st.markdown("""
                    <div class="info-box">
                        💡 <strong>Encoding Issues?</strong><br>
                        • Open the file in Excel/Notepad<br>
                        • Save As → Choose 'CSV UTF-8' or 'Text (UTF-8)'<br>
                        • Or install chardet: <code>pip install chardet</code>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif ftype == 'dataframe':
                st.markdown(f"""
                <div class="success-box">
                    ✅ Successfully loaded <strong>{len(content)} rows</strong>
                    {f'<br>📝 Encoding: {encoding_info}' if encoding_info else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Preview
                with st.expander("📋 Preview Data", expanded=True):
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
                        ✅ Text column detected: <strong>{col}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    max_rows = st.slider("Maximum rows to analyze:", 10, 500, 100)
                    
                    if st.button("▶️ Start Analysis", type="primary", use_container_width=True):
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
                                <div class="metric-card">
                                    <div class="metric-value">{len(results)}</div>
                                    <div class="metric-label">Analyzed</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #10b981;">{pos}</div>
                                    <div class="metric-label">Positive</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color: #ef4444;">{len(results)-pos}</div>
                                    <div class="metric-label">Negative</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(create_distribution_pie(df), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(df, use_container_width=True, height=400)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ No valid text found to analyze")
                else:
                    st.error("❌ No text column detected in the file")
                    st.info(f"Available columns: {', '.join(content.columns)}")
                    selected_col = st.selectbox("Manually select text column:", content.columns)
                    if st.button("Use Selected Column"):
                        col = selected_col
                        st.rerun()
            
            elif ftype == 'text':
                st.markdown(f"""
                <div class="success-box">
                    ✅ Found <strong>{len(content)} text items</strong>
                    {f'<br>📝 Encoding: {encoding_info}' if encoding_info else ''}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 Preview", expanded=True):
                    for i, item in enumerate(content[:5]):
                        st.text(f"{i+1}. {item[:100]}...")
                
                max_items = st.slider("Maximum items to analyze:", 10, 500, min(100, len(content)))
                
                if st.button("▶️ Start Analysis", type="primary", use_container_width=True):
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
                            <div class="metric-card">
                                <div class="metric-value">{len(results)}</div>
                                <div class="metric-label">Analyzed</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: #10b981;">{pos}</div>
                                <div class="metric-label">Positive</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: #ef4444;">{len(results)-pos}</div>
                                <div class="metric-label">Negative</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(create_distribution_pie(df), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True, height=400)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No valid text found to analyze")
    
    # ========== TAB 4: ANALYTICS ==========
    with tab4:
        st.markdown("### 📊 Analytics Dashboard")
        st.caption("Executive summary of all your analyses")
        
        if st.session_state.analysis_history:
            df = pd.DataFrame(st.session_state.analysis_history)
            
            # Top metrics
            total = len(df)
            pos = len(df[df['sentiment']=='Positive'])
            avg = df['confidence'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">Total Analyses</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #10b981;">{pos/total:.1%}</div>
                    <div class="metric-label">Positive Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg:.1%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### 📜 Recent Activity")
            
            for _, row in df.tail(10).sort_values('timestamp', ascending=False).iterrows():
                emoji = "😊" if row['sentiment'] == "Positive" else "😠"
                sentiment_color = "#10b981" if row['sentiment'] == "Positive" else "#ef4444"
                
                st.markdown(f"""
                <div class="activity-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.5rem;">{emoji}</span>
                            <strong style="color: {sentiment_color}; font-size: 1.1rem;">{row['sentiment']}</strong>
                            <span style="color: #6b7280; margin-left: 0.5rem;">({row['confidence']:.1%})</span>
                        </div>
                    </div>
                    <div class="activity-text">{row['text']}</div>
                    <div class="activity-meta">
                        🌍 {row.get('language', 'en').upper()} • 📅 {row['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="premium-card" style="text-align: center; padding: 3rem;">
                <h3>📭 No Analysis History</h3>
                <p style="color: #6b7280; margin-top: 1rem;">
                    Start analyzing text to see your activity dashboard populate with insights
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()