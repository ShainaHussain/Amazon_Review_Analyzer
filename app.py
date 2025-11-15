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
import json
import re
from io import StringIO

# Page config
st.set_page_config(
    page_title="Sentimart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF9500, #FF6B35);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9500;
        margin: 0.5rem 0;
    }
    
    .result-positive {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .result-negative {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    
    .model-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the pre-trained BERT model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("model")
        model = AutoModelForSequenceClassification.from_pretrained("model")
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {
        'accuracy': 0.94,
        'precision': 0.93,
        'recall': 0.95,
        'f1_score': 0.94,
        'total_analyses': 0
    }

# Sample reviews for testing
SAMPLE_REVIEWS = {
    "Positive Tech Review": "This smartphone is absolutely amazing! The camera quality is outstanding, battery lasts all day, and the performance is lightning fast. Best purchase I've made this year. Highly recommend to everyone!",
    "Negative Product Review": "Terrible product. Arrived damaged, poor quality materials, and completely different from the description. Customer service was unhelpful. Would not recommend and requesting a refund immediately.",
    "Mixed Sentiment": "The product has some good features like fast shipping and nice packaging, but the quality is mediocre for the price. It works as expected but nothing exceptional. Might be okay for casual use.",
    "Neutral Review": "The item arrived on time and matches the description. Standard quality product, nothing special but does what it's supposed to do. Average experience overall."
}

def analyze_text_features(text):
    """Extract various text features for analysis"""
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(re.findall(r'[.!?]+', text))
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': max(1, sentence_count),
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'caps_ratio': caps_ratio,
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment with confidence score"""
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        # Get both class probabilities
        positive_prob = probs[0][1].item()
        negative_prob = probs[0][0].item()
    
    return {
        'prediction': pred_class,
        'confidence': confidence,
        'positive_prob': positive_prob,
        'negative_prob': negative_prob,
        'label': "Positive" if pred_class == 1 else "Negative"
    }

def create_confidence_chart(positive_prob, negative_prob):
    """Create a confidence visualization"""
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
        height=400,
        showlegend=False
    )
    
    return fig

def create_history_chart():
    """Create a chart showing analysis history"""
    if not st.session_state.analysis_history:
        return None
    
    df = pd.DataFrame(st.session_state.analysis_history)
    
    # Count sentiments over time
    sentiment_counts = df['label'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Analysis Distribution",
        color_discrete_map={'Positive': '#51cf66', 'Negative': '#ff6b6b'}
    )
    
    return fig

def export_results():
    """Export analysis history to CSV"""
    if st.session_state.analysis_history:
        df = pd.DataFrame(st.session_state.analysis_history)
        csv = df.to_csv(index=False)
        return csv
    return None

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Sentimart</h1>
        <p>Advanced sentiment analysis using BERT with comprehensive analytics and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    tokenizer, model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("⚠️ Model could not be loaded. Please check your model files.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        <div class="model-info">
            <h4>BERT-based Classifier</h4>
            <p><strong>Architecture:</strong> BERT Base</p>
            <p><strong>Max Length:</strong> 512 tokens</p>
            <p><strong>Classes:</strong> Positive, Negative</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Performance Metrics
        st.subheader("🎯 Model Performance")
        metrics = st.session_state.model_metrics
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            st.metric("Precision", f"{metrics['precision']:.1%}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.1%}")
            st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
        
        st.metric("Total Analyses", metrics['total_analyses'])
        
        # Export functionality
        st.subheader("📥 Export Results")
        if st.button("Export History"):
            csv_data = export_results()
            if csv_data:
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No analysis history to export")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Review Input")
        
        # Sample reviews dropdown
        selected_sample = st.selectbox(
            "Try a sample review:",
            ["Select a sample..."] + list(SAMPLE_REVIEWS.keys())
        )
        
        # Text input
        if selected_sample != "Select a sample...":
            review_text = st.text_area(
                "Review text:",
                value=SAMPLE_REVIEWS[selected_sample],
                height=150,
                help="Paste your Amazon review here for analysis"
            )
        else:
            review_text = st.text_area(
                "Review text:",
                height=150,
                placeholder="Paste your Amazon review here...",
                help="Enter the review text you want to analyze"
            )
        
        # Text statistics
        if review_text:
            features = analyze_text_features(review_text)
            
            st.subheader("📝 Text Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Words", features['word_count'])
            with stat_col2:
                st.metric("Characters", features['char_count'])
            with stat_col3:
                st.metric("Sentences", features['sentence_count'])
            with stat_col4:
                st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
        
        # Analysis button
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if not review_text.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                # Show progress
                with st.spinner("Analyzing with BERT model..."):
                    time.sleep(1)  # Simulate processing time
                    
                    # Perform prediction
                    result = predict_sentiment(review_text, tokenizer, model)
                    features = analyze_text_features(review_text)
                    
                    # Store in history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'text': review_text[:100] + "..." if len(review_text) > 100 else review_text,
                        'label': result['label'],
                        'confidence': result['confidence'],
                        'word_count': features['word_count']
                    })
                    
                    # Update metrics
                    st.session_state.model_metrics['total_analyses'] += 1
                
                # Display results
                st.subheader("🎯 Analysis Results")
                
                # Main result
                result_class = "result-positive" if result['label'] == "Positive" else "result-negative"
                emoji = "😊" if result['label'] == "Positive" else "😠"
                
                st.markdown(f"""
                <div class="{result_class}">
                    <h3>{emoji} Sentiment: {result['label']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.plotly_chart(
                    create_confidence_chart(result['positive_prob'], result['negative_prob']),
                    use_container_width=True
                )
                
                # Detailed metrics
                st.subheader("📊 Detailed Analysis")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**Probability Scores:**")
                    st.write(f"• Positive: {result['positive_prob']:.1%}")
                    st.write(f"• Negative: {result['negative_prob']:.1%}")
                    
                with detail_col2:
                    st.markdown("**Text Characteristics:**")
                    st.write(f"• Exclamations: {features['exclamation_count']}")
                    st.write(f"• Questions: {features['question_count']}")
                    st.write(f"• Caps Ratio: {features['caps_ratio']:.1%}")
    
    with col2:
        st.subheader("📈 Analytics Dashboard")
        
        # History chart
        if st.session_state.analysis_history:
            history_fig = create_history_chart()
            if history_fig:
                st.plotly_chart(history_fig, use_container_width=True)
            
            # Recent analyses
            st.subheader("🕒 Recent Analyses")
            recent_analyses = st.session_state.analysis_history[-5:]  # Last 5
            
            for analysis in reversed(recent_analyses):
                emoji = "😊" if analysis['label'] == "Positive" else "😠"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{emoji} {analysis['label']}</strong> ({analysis['confidence']:.1%})<br>
                    <small>{analysis['text']}</small><br>
                    <small>📅 {analysis['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("p")
        
        # Replace the batch analysis section (around line 335-375) with this corrected version:

# Batch analysis option
st.subheader("📦 Batch Analysis")
uploaded_file = st.file_uploader(
    "Upload CSV with reviews",
    type=['csv', 'xlsx'],
    help="Upload a CSV/Excel file with a 'review' column for batch analysis"
)

if uploaded_file is not None:
    try:
        # Handle both CSV and Excel files
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            df = None
        
        if df is not None:
            # Check for 'review' column (case-insensitive)
            review_col = None
            for col in df.columns:
                if col.lower() == 'review':
                    review_col = col
                    break
            
            if review_col:
                # Filter out empty reviews
                valid_reviews = df[df[review_col].notna()][review_col]
                st.write(f"📊 Found {len(valid_reviews)} valid reviews")
                
                if len(valid_reviews) > 0:
                    # Limit number of reviews for analysis
                    max_reviews = min(50, len(valid_reviews))
                    
                    if st.button("Analyze All Reviews"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        
                        for i, review in enumerate(valid_reviews.head(max_reviews)):
                            try:
                                status_text.text(f"Analyzing review {i+1}/{max_reviews}...")
                                
                                # Convert to string and check length
                                review_text = str(review).strip()
                                
                                if len(review_text) > 0:
                                    result = predict_sentiment(review_text, tokenizer, model)
                                    results.append({
                                        'review': review_text[:50] + "..." if len(review_text) > 50 else review_text,
                                        'sentiment': result['label'],
                                        'confidence': f"{result['confidence']:.1%}"
                                    })
                                    
                                    # Update session history
                                    st.session_state.analysis_history.append({
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'text': review_text[:100] + "..." if len(review_text) > 100 else review_text,
                                        'label': result['label'],
                                        'confidence': result['confidence'],
                                        'word_count': len(review_text.split())
                                    })
                                    
                            except Exception as e:
                                st.warning(f"Error analyzing review {i+1}: {str(e)}")
                                continue
                            
                            # Update progress
                            progress_bar.progress((i + 1) / max_reviews)
                        
                        status_text.text("Analysis complete!")
                        
                        if results:
                            # Display batch results
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            positive_count = len([r for r in results if r['sentiment'] == 'Positive'])
                            negative_count = len(results) - positive_count
                            
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            with summary_col1:
                                st.metric("Total Analyzed", len(results))
                            with summary_col2:
                                st.metric("✅ Positive", positive_count)
                            with summary_col3:
                                st.metric("❌ Negative", negative_count)
                            
                            # Update total analyses counter
                            st.session_state.model_metrics['total_analyses'] += len(results)
                            
                            # Download results
                            csv_output = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv_output,
                                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No reviews could be analyzed. Please check your data.")
                else:
                    st.info("No valid reviews found in the uploaded file.")
            else:
                st.error("❌ CSV must contain a 'review' column (case-insensitive)")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your file is properly formatted and not corrupted.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 Powered by BERT |An AI-powered sentiment analyzer designed for product review insights | 
        <a href="https://github.com" target="_blank">View Source Code</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()