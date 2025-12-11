"""
Sentimart Pro - Advanced Multi-Source Sentiment Analysis
Features:
- Real API integration (YouTube)
- PDF support
- Word clouds
- Emotion detection
- Multi-language support
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

# Optional imports with error handling
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ Install PyPDF2 for PDF support: pip install PyPDF2")

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

# Page config
st.set_page_config(
    page_title="Sentimart Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .result-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
    }
    .result-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 600;
        background: #e3f2fd;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
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
            device=-1  # CPU
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
# SESSION STATE
# ============================================================================

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0

# ============================================================================
# PREDICTION FUNCTIONS
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
# API EXTRACTION FUNCTIONS
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
        # Demo mode
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
# FILE READING
# ============================================================================

def read_pdf(file):
    if not PDF_AVAILABLE:
        return None, "PyPDF2 not installed"
    try:
        pdf = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
        return paragraphs, None
    except Exception as e:
        return None, str(e)

def read_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if ext == 'csv':
            return pd.read_csv(uploaded_file), 'dataframe', None
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file), 'dataframe', None
        elif ext == 'txt':
            content = uploaded_file.read().decode('utf-8')
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            return lines, 'text', None
        elif ext == 'pdf':
            paragraphs, error = read_pdf(uploaded_file)
            return paragraphs, 'text' if paragraphs else 'error', error
        else:
            return None, 'unsupported', "Unsupported format"
    except Exception as e:
        return None, 'error', str(e)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_confidence_chart(pos, neg):
    fig = go.Figure(data=[
        go.Bar(x=['Negative', 'Positive'], y=[neg, pos],
               marker_color=['#ff6b6b', '#51cf66'],
               text=[f'{neg:.1%}', f'{pos:.1%}'], textposition='auto')
    ])
    fig.update_layout(title="Confidence", height=300, showlegend=False, yaxis=dict(tickformat='.0%'))
    return fig

def create_emotion_chart(emotions):
    names = list(emotions.keys())
    scores = list(emotions.values())
    fig = go.Figure(data=[
        go.Bar(x=names, y=scores, marker_color='#667eea',
               text=[f'{s:.1%}' for s in scores], textposition='auto')
    ])
    fig.update_layout(title="Emotions", height=300, yaxis=dict(tickformat='.0%'))
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
    colors = ['#51cf66' if l == 'Positive' else '#ff6b6b' for l in counts.index]
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.4, marker_colors=colors)])
    fig.update_layout(title="Distribution", height=300)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Sentimart Pro</h1>
        <p>Advanced Multi-Source Sentiment & Emotion Analysis</p>
        <p style="font-size:0.85em;opacity:0.9">✨ YouTube API • 📄 PDF • ☁️ Word Clouds • 😊 Emotions • 🌍 Multi-Language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    tokenizer, model, loaded = load_sentiment_model()
    emotion_clf, emotion_ok = load_emotion_model()
    ml_tok, ml_model, ml_ok = load_multilingual_model()
    
    if not loaded:
        st.error("⚠️ Model failed to load")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🔑 API Keys"):
            youtube_key = st.text_input("AIzaSyBH2RJD9X-lBWdbN34_pIvd14WtfbYSftI", type="password", help="Optional for real data")
        
        with st.expander("🤖 Models"):
            st.write("Sentiment:", "✅" if loaded else "❌")
            st.write("Emotion:", "✅" if emotion_ok else "❌")
            st.write("Multilingual:", "✅" if ml_ok else "❌")
            st.write("PDF:", "✅" if PDF_AVAILABLE else "❌")
        
        st.metric("Total Analyses", st.session_state.total_analyses)
        
        if st.session_state.batch_results is not None:
            csv = st.session_state.batch_results.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, 
                             f"results_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single", "🔗 YouTube", "📦 File", "📊 Analytics"])
    
    # TAB 1: Single Analysis
    with tab1:
        st.subheader("✍️ Single Text Analysis")
        
        text = st.text_area("Enter text:", height=150, placeholder="Your review here...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            do_emotion = st.checkbox("Emotions", value=emotion_ok)
        with col2:
            do_ml = st.checkbox("Multi-lang", value=ml_ok)
        with col3:
            show_wc = st.checkbox("Word cloud", value=False)
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Enter text")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_sentiment(text, tokenizer, model)
                    lang = detect_language(text)
                    
                    emotion_res = predict_emotions(text, emotion_clf) if do_emotion and emotion_ok else None
                    ml_res = predict_multilingual(text, ml_tok, ml_model) if do_ml and ml_ok and lang != 'en' else None
                    
                    st.session_state.total_analyses += 1
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'text': text[:80], 'sentiment': result['label'],
                        'confidence': result['confidence'], 'language': lang
                    })
                
                cls = "result-positive" if result['label'] == "Positive" else "result-negative"
                emoji = "😊" if result['label'] == "Positive" else "😠"
                
                st.markdown(f"""
                <div class="{cls}">
                    <h3>{emoji} {result['label']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%} | <strong>Language:</strong> {lang.upper()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_confidence_chart(result['positive_prob'], result['negative_prob']), use_container_width=True)
                
                if emotion_res:
                    with col2:
                        st.plotly_chart(create_emotion_chart(emotion_res['emotions']), use_container_width=True)
                
                if ml_res:
                    st.info(f"🌍 Multilingual: {ml_res['label']} ({ml_res['rating']}/5) - {ml_res['confidence']:.1%}")
                
                if show_wc:
                    wc = create_wordcloud([text])
                    if wc:
                        st.pyplot(wc)
    
    # TAB 2: YouTube Analysis
    with tab2:
        st.subheader("🔗 YouTube Comments Analysis")
        
        url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
        
        col1, col2 = st.columns(2)
        with col1:
            max_items = st.slider("Max comments:", 10, 100, 30)
        with col2:
            show_wc_url = st.checkbox("Word cloud", value=True, key="wc_url")
        
        if st.button("🚀 Fetch & Analyze", type="primary"):
            if not url:
                st.warning("Enter YouTube URL")
            elif 'youtube' not in url and 'youtu.be' not in url:
                st.error("Please enter a valid YouTube URL")
            else:
                with st.spinner("Fetching YouTube comments..."):
                    comments, mode = fetch_youtube_comments(url, youtube_key if 'youtube_key' in locals() else None, max_items)
                    
                    if isinstance(mode, str) and mode in ['demo', 'api']:
                        st.info(f"Mode: {mode.upper()}")
                    else:
                        st.error(f"Error: {mode}")
                        st.stop()
                    
                    if comments:
                        st.success(f"✅ Fetched {len(comments)} comments")
                        
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
                            
                            col1, col2, col3 = st.columns(3)
                            pos = len(df[df['sentiment']=='Positive'])
                            col1.metric("Total", len(df))
                            col2.metric("Positive", pos)
                            col3.metric("Negative", len(df)-pos)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(create_distribution_pie(df), use_container_width=True)
                            
                            if show_wc_url:
                                with col2:
                                    wc = create_wordcloud(df['text'].tolist())
                                    if wc:
                                        st.pyplot(wc)
                            
                            st.dataframe(df, use_container_width=True)
    
    # TAB 3: File Upload
    with tab3:
        st.subheader("📦 Batch File Analysis")
        st.info("Supports: CSV, Excel, TXT, PDF")
        
        file = st.file_uploader("Upload file:", type=['csv', 'xlsx', 'txt', 'pdf'])
        
        if file:
            content, ftype, error = read_file(file)
            
            if error:
                st.error(error)
            elif ftype == 'dataframe':
                st.success(f"Loaded {len(content)} rows")
                st.dataframe(content.head(), use_container_width=True)
                
                col = None
                for c in content.columns:
                    if c.lower() in ['review', 'text', 'comment']:
                        col = c
                        break
                
                if col:
                    st.info(f"Using column: **{col}**")
                    
                    if st.button("▶️ Analyze All", type="primary"):
                        valid = content[content[col].notna()][col]
                        results = []
                        prog = st.progress(0)
                        
                        for i, txt in enumerate(valid.head(200)):
                            r = predict_sentiment(str(txt), tokenizer, model)
                            results.append({'review': str(txt)[:50], 'sentiment': r['label'], 
                                          'confidence': r['confidence']})
                            prog.progress((i+1)/min(len(valid), 200))
                        
                        df = pd.DataFrame(results)
                        st.session_state.batch_results = df
                        st.session_state.total_analyses += len(results)
                        
                        st.success(f"✅ Analyzed {len(results)}")
                        st.dataframe(df, use_container_width=True)
                else:
                    st.error("No 'review' column found")
            
            elif ftype == 'text':
                st.success(f"Found {len(content)} items")
                
                if st.button("▶️ Analyze", type="primary"):
                    results = []
                    prog = st.progress(0)
                    
                    for i, txt in enumerate(content[:200]):
                        r = predict_sentiment(txt, tokenizer, model)
                        results.append({'text': txt[:50], 'sentiment': r['label'], 
                                      'confidence': r['confidence']})
                        prog.progress((i+1)/min(len(content), 200))
                    
                    df = pd.DataFrame(results)
                    st.session_state.batch_results = df
                    st.dataframe(df, use_container_width=True)
    
    # TAB 4: Analytics
    with tab4:
        st.subheader("📊 Analytics Dashboard")
        
        if st.session_state.analysis_history:
            df = pd.DataFrame(st.session_state.analysis_history)
            
            col1, col2, col3 = st.columns(3)
            total = len(df)
            pos = len(df[df['sentiment']=='Positive'])
            avg = df['confidence'].mean()
            
            col1.metric("Total", total)
            col2.metric("Positive Rate", f"{pos/total:.1%}")
            col3.metric("Avg Confidence", f"{avg:.1%}")
            
            st.subheader("Recent Activity")
            for _, row in df.tail(10).sort_values('timestamp', ascending=False).iterrows():
                emoji = "😊" if row['sentiment'] == "Positive" else "😠"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{emoji} {row['sentiment']}</strong> ({row['confidence']:.1%})<br>
                    <small>{row['text']}</small><br>
                    <small>🌍 {row.get('language', 'en').upper()} | 📅 {row['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No history yet")

if __name__ == "__main__":
    main()