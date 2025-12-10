import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import re
from urllib.parse import urlparse, parse_qs

# Page config
st.set_page_config(
    page_title="Sentimart Advanced",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with gradient theme
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL & SESSION STATE
# ============================================================================

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("model")
        model = AutoModelForSequenceClassification.from_pretrained("model")
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        positive_prob = probs[0][1].item()
        negative_prob = probs[0][0].item()
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'positive_prob': positive_prob,
        'negative_prob': negative_prob,
        'label': "Positive" if pred_class == 1 else "Negative"
    }

def detect_url_type(url):
    url_lower = url.lower()
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'reddit.com' in url_lower:
        return 'reddit'
    elif 'amazon' in url_lower:
        return 'amazon'
    return 'unknown'

def extract_youtube_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
    return None

def fetch_demo_comments(platform, count=50):
    """Generate demo comments for different platforms"""
    youtube_comments = [
        "This video is absolutely amazing! Great content and very informative.",
        "Thanks for sharing this. Really helpful tutorial!",
        "Not sure about this approach. Seems too complicated for beginners.",
        "Excellent explanation! Subscribed!",
        "Could you make a follow-up video on advanced topics?",
        "This didn't work for me. Getting errors.",
        "Best tutorial on this topic I've found. Thank you!",
        "The quality could be better but good content overall.",
        "Waste of time. Nothing new here.",
        "Perfect timing! I was just looking for this information.",
    ]
    
    reddit_comments = [
        "This is exactly what I needed! Thanks for posting.",
        "Not convinced this is the best approach.",
        "Can confirm, this worked perfectly for me!",
        "Terrible advice. Please don't follow this.",
        "Great post! Saved for later reference.",
        "Has anyone else tried this? Results?",
        "This should be upvoted more. Quality content.",
        "Downvoted. This is misleading information.",
    ]
    
    amazon_reviews = [
        "Amazing product! Exceeded my expectations in every way.",
        "Good quality for the price. Would recommend.",
        "Decent product but shipping was slow.",
        "Not as described. Very disappointed.",
        "Terrible quality. Broke after one use.",
        "Exactly what I was looking for! Fast shipping too.",
        "Works well but instructions could be clearer.",
        "Five stars! Best purchase this year.",
        "Returned it. Complete waste of money.",
        "Perfect for my needs. Happy with purchase.",
    ]
    
    if platform == 'youtube':
        return youtube_comments[:count]
    elif platform == 'reddit':
        return reddit_comments[:count]
    elif platform == 'amazon':
        return amazon_reviews[:count]
    return []

def read_file_content(uploaded_file):
    """Read content from uploaded files"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            return df, 'dataframe'
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            return df, 'dataframe'
        
        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
            return content, 'text'
        
        elif file_extension == 'json':
            content = uploaded_file.read().decode('utf-8')
            import json
            data = json.loads(content)
            return data, 'json'
        
        else:
            return None, 'unsupported'
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, 'error'

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_confidence_chart(positive_prob, negative_prob):
    fig = go.Figure(data=[
        go.Bar(
            x=['Negative', 'Positive'],
            y=[negative_prob, positive_prob],
            marker_color=['#ff6b6b', '#51cf66'],
            text=[f'{negative_prob:.1%}', f'{positive_prob:.1%}'],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Sentiment Confidence Scores",
        xaxis_title="Sentiment",
        yaxis_title="Probability",
        yaxis=dict(tickformat='.0%'),
        height=350,
        showlegend=False
    )
    return fig

def create_batch_distribution_chart(results_df):
    sentiment_counts = results_df['sentiment'].value_counts()
    fig = go.Figure(data=[
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=['#51cf66' if label == 'Positive' else '#ff6b6b' 
                          for label in sentiment_counts.index]
        )
    ])
    fig.update_layout(title="Sentiment Distribution", height=350)
    return fig

def create_confidence_histogram(results_df):
    fig = go.Figure(data=[
        go.Histogram(
            x=results_df['confidence'],
            nbinsx=20,
            marker_color='#667eea',
            opacity=0.7
        )
    ])
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=350
    )
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Sentimart Advanced</h1>
        <p>Multi-Source Sentiment Analysis | Files • URLs • Batch Processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    tokenizer, model, model_loaded = load_model()
    if not model_loaded:
        st.error("⚠️ Model could not be loaded.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard")
        st.metric("Total Analyses", st.session_state.total_analyses)
        
        st.markdown("---")
        st.subheader("💡 Quick Tips")
        st.info("""
        **URL Analysis:**
        - YouTube: Paste video URL
        - Reddit: Paste post URL  
        - Amazon: Paste product URL
        
        **File Upload:**
        - CSV with 'review' column
        - Excel files supported
        - TXT files with one review per line
        """)
        
        if st.session_state.batch_results is not None:
            st.markdown("---")
            st.subheader("📥 Export Results")
            csv_data = st.session_state.batch_results.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single Text", 
        "🔗 URL Analysis", 
        "📦 File Upload",
        "📊 Analytics"
    ])
    
    # ========================================================================
    # TAB 1: SINGLE TEXT ANALYSIS
    # ========================================================================
    with tab1:
        st.subheader("✍️ Analyze Single Review")
        
        review_text = st.text_area(
            "Enter your review:",
            height=200,
            placeholder="Paste your review text here...",
            help="Enter any product review, comment, or feedback"
        )
        
        if review_text:
            word_count = len(review_text.split())
            char_count = len(review_text)
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", word_count)
            col2.metric("Characters", char_count)
            col3.metric("Sentences", len(re.findall(r'[.!?]+', review_text)))
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if not review_text.strip():
                st.warning("Please enter some text")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_sentiment(review_text, tokenizer, model)
                    st.session_state.total_analyses += 1
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'text': review_text[:100],
                        'sentiment': result['label'],
                        'confidence': result['confidence']
                    })
                
                result_class = "result-positive" if result['label'] == "Positive" else "result-negative"
                emoji = "😊" if result['label'] == "Positive" else "😠"
                
                st.markdown(f"""
                <div class="{result_class}">
                    <h3>{emoji} Sentiment: {result['label']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(
                    create_confidence_chart(result['positive_prob'], result['negative_prob']),
                    use_container_width=True
                )
    
    # ========================================================================
    # TAB 2: URL ANALYSIS
    # ========================================================================
    with tab2:
        st.subheader("🔗 Analyze Content from URLs")
        st.info("📌 Supports: YouTube, Reddit, Amazon (Demo mode - no API keys required)")
        
        url_input = st.text_input(
            "Enter URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video, Reddit post, or Amazon product URL"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_items = st.slider("Items to analyze:", 10, 100, 30)
        with col2:
            if url_input:
                url_type = detect_url_type(url_input)
                st.info(f"Detected: **{url_type.upper()}**")
        
        if st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True):
            if not url_input:
                st.warning("Please enter a URL")
            else:
                url_type = detect_url_type(url_input)
                
                if url_type == 'unknown':
                    st.error("Unsupported URL. Use YouTube, Reddit, or Amazon URLs.")
                else:
                    with st.spinner(f"Fetching {url_type} content..."):
                        # Demo mode - fetch sample comments
                        comments = fetch_demo_comments(url_type, max_items)
                        
                        if not comments:
                            st.error("No content found")
                        else:
                            st.success(f"✅ Fetched {len(comments)} items")
                            
                            # Analyze sentiments
                            progress_bar = st.progress(0)
                            results = []
                            
                            for i, comment in enumerate(comments):
                                if len(comment.strip()) > 0:
                                    result = predict_sentiment(comment, tokenizer, model)
                                    results.append({
                                        'text': comment[:80] + "..." if len(comment) > 80 else comment,
                                        'sentiment': result['label'],
                                        'confidence': result['confidence']
                                    })
                                progress_bar.progress((i + 1) / len(comments))
                            
                            if results:
                                results_df = pd.DataFrame(results)
                                st.session_state.batch_results = results_df
                                st.session_state.total_analyses += len(results)
                                
                                # Summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                positive = len(results_df[results_df['sentiment'] == 'Positive'])
                                negative = len(results_df) - positive
                                avg_conf = results_df['confidence'].mean()
                                
                                col1.metric("Total", len(results_df))
                                col2.metric("✅ Positive", positive)
                                col3.metric("❌ Negative", negative)
                                col4.metric("Avg Confidence", f"{avg_conf:.1%}")
                                
                                # Visualizations
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(create_batch_distribution_chart(results_df), use_container_width=True)
                                with col2:
                                    st.plotly_chart(create_confidence_histogram(results_df), use_container_width=True)
                                
                                # Show results table
                                st.subheader("📋 Detailed Results")
                                st.dataframe(results_df, use_container_width=True)
    
    # ========================================================================
    # TAB 3: FILE UPLOAD
    # ========================================================================
    with tab3:
        st.subheader("📦 Batch File Analysis")
        st.info("📌 Supported: CSV, Excel, TXT, JSON")
        
        uploaded_file = st.file_uploader(
            "Upload your file:",
            type=['csv', 'xlsx', 'xls', 'txt', 'json'],
            help="CSV/Excel: Must have 'review' column | TXT: One review per line"
        )
        
        if uploaded_file is not None:
            content, file_type = read_file_content(uploaded_file)
            
            if file_type == 'dataframe':
                st.success(f"✅ Loaded: {len(content)} rows")
                st.dataframe(content.head(), use_container_width=True)
                
                # Find review column
                review_col = None
                for col in content.columns:
                    if col.lower() in ['review', 'text', 'comment', 'feedback']:
                        review_col = col
                        break
                
                if review_col:
                    st.info(f"Using column: **{review_col}**")
                    
                    if st.button("▶️ Analyze All", type="primary", use_container_width=True):
                        valid_reviews = content[content[review_col].notna()][review_col]
                        max_reviews = min(200, len(valid_reviews))
                        
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, review in enumerate(valid_reviews.head(max_reviews)):
                            text = str(review).strip()
                            if len(text) > 0:
                                result = predict_sentiment(text, tokenizer, model)
                                results.append({
                                    'review': text[:60] + "..." if len(text) > 60 else text,
                                    'sentiment': result['label'],
                                    'confidence': result['confidence']
                                })
                            progress_bar.progress((i + 1) / max_reviews)
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.session_state.batch_results = results_df
                            st.session_state.total_analyses += len(results)
                            
                            st.success(f"✅ Analyzed {len(results)} reviews")
                            
                            # Summary
                            col1, col2, col3 = st.columns(3)
                            positive = len(results_df[results_df['sentiment'] == 'Positive'])
                            col1.metric("Total", len(results_df))
                            col2.metric("✅ Positive", positive)
                            col3.metric("❌ Negative", len(results_df) - positive)
                            
                            # Chart
                            st.plotly_chart(create_batch_distribution_chart(results_df), use_container_width=True)
                            
                            # Results table
                            st.dataframe(results_df, use_container_width=True)
                else:
                    st.error("No review column found. Please ensure your file has a 'review', 'text', or 'comment' column.")
                    st.write("Available columns:", list(content.columns))
            
            elif file_type == 'text':
                reviews = [line.strip() for line in content.split('\n') if line.strip()]
                st.success(f"✅ Found {len(reviews)} reviews")
                
                if st.button("▶️ Analyze All", type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, review in enumerate(reviews[:200]):
                        result = predict_sentiment(review, tokenizer, model)
                        results.append({
                            'review': review[:60] + "..." if len(review) > 60 else review,
                            'sentiment': result['label'],
                            'confidence': result['confidence']
                        })
                        progress_bar.progress((i + 1) / min(len(reviews), 200))
                    
                    results_df = pd.DataFrame(results)
                    st.session_state.batch_results = results_df
                    st.session_state.total_analyses += len(results)
                    
                    st.dataframe(results_df, use_container_width=True)
    
    # ========================================================================
    # TAB 4: ANALYTICS
    # ========================================================================
    with tab4:
        st.subheader("📊 Analytics Dashboard")
        
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            col1, col2, col3 = st.columns(3)
            total = len(history_df)
            positive = len(history_df[history_df['sentiment'] == 'Positive'])
            avg_conf = history_df['confidence'].mean()
            
            col1.metric("Total Analyses", total)
            col2.metric("Positive Rate", f"{positive/total:.1%}")
            col3.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            # Recent history
            st.subheader("🕒 Recent Analyses")
            recent = history_df.tail(10).sort_values('timestamp', ascending=False)
            
            for _, row in recent.iterrows():
                emoji = "😊" if row['sentiment'] == "Positive" else "😠"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{emoji} {row['sentiment']}</strong> ({row['confidence']:.1%})<br>
                    <small>{row['text']}...</small><br>
                    <small>📅 {row['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analysis history yet. Start analyzing to see statistics!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 Powered by BERT | Advanced Multi-Source Sentiment Analyzer
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()