# 🛒 Sentimart  
**A Research-Driven, Multi-Source Sentiment & Emotion Intelligence System**

Sentimart is an actively evolving sentiment intelligence platform designed to analyze opinions from heterogeneous, real-world data sources. The system integrates transformer based NLP models, emotion classification, multilingual understanding, and visual analytics to move beyond basic polarity detection toward deeper emotional insight.

This project is not a static demo , it is a **living system under continuous development**, focused on experimentation, scalability, and research-oriented extensions.

## 🚀 Live Application
👉 https://asentimart.streamlit.app/

## 🎯 Motivation & Vision
Textual sentiment in the real world is **rarely binary** and **rarely clean**. Reviews, comments, and feedback often carry:
- mixed emotions,
- language variation,
- noisy encodings,
- and contextual ambiguity.

Sentimart is built to address these realities by progressively evolving from **simple sentiment classification** to **multi-dimensional emotional intelligence** across platforms and formats.


## 🧠 Core Capabilities

### 📝 Single Text Intelligence
- Transformer-based sentiment classification
- Confidence-aware predictions
- Automatic language detection
- Emotion profiling
- Optional word-cloud based lexical insight

### 🔗 YouTube Opinion Mining
- Real YouTube Data API integration
- Scalable batch comment processing
- Sentiment distribution analytics
- Demo fallback mode for reproducibility
- Visual and tabular insights

### 📦 Document & Dataset Analysis
- Supports **CSV, Excel, TXT, PDF**
- Robust handling of real-world encoding issues
- Automatic text column inference
- High-volume batch sentiment aggregation

### 😊 Emotion Understanding
- Multi-label emotion classification
- Emotion confidence visualization
- Designed for extension toward compound emotions

### 🌍 Multilingual Reasoning
- Multilingual transformer models for non-English inputs
- Rating-based sentiment interpretation
- Language-aware prediction routing

### 📊 Analytical Tracking
- Session-level analytics
- Confidence trend monitoring
- Historical interaction logging

## 🖼️ Application Interface

<img width="752" height="439" alt="image" src="https://github.com/user-attachments/assets/76cc01db-1e59-4249-9be8-a1e3971d37fe" />
<img width="748" height="455" alt="image" src="https://github.com/user-attachments/assets/09b5e881-570b-4ab7-962c-889caef297a3" />
<img width="767" height="442" alt="image" src="https://github.com/user-attachments/assets/3cb10d42-ed2d-4d11-86f8-05cddeb0b659" />
<img width="765" height="443" alt="image" src="https://github.com/user-attachments/assets/ab53cf24-e047-4af7-b05a-0f02f47c4164" />
<img width="749" height="437" alt="image" src="https://github.com/user-attachments/assets/07e985b5-459c-4931-8179-37f34d0c43e3" />
<img width="926" height="427" alt="image" src="https://github.com/user-attachments/assets/c3062245-cd26-4376-9cc5-15ee7014e7da" />


## 🛠️ Technology Stack

- Frontend / UI: Streamlit
- Core NLP: Hugging Face Transformers, PyTorch
- Emotion Model: j-hartmann/emotion-english-distilroberta-base
- Multilingual Sentiment: nlptown/bert-base-multilingual-uncased-sentiment
- Data Processing: Pandas, NumPy
- Visualization: Plotly, Matplotlib, WordCloud
- APIs: YouTube Data API v3
- Utilities: PyPDF2, chardet, langdetect

## ⚙️ System Workflow

- Input ingestion (text, URL, or document)
- Encoding normalization and language detection
- Dynamic model selection based on input characteristics
- Sentiment and emotion inference
- Visualization, aggregation, and analytics tracking

## 📂 Supported Input Sources
Source          	Status

Plain Text      	✅

YouTube	          ✅

CSV             	✅

Excel           	✅

TXT             	✅

PDF             	✅

## 🔐 API Integration

- YouTube analysis runs in demo mode by default
- Real comment extraction is enabled via user-provided API keys
- API usage is optional and modular

## ▶️ Local Execution

clone https://github.com/ShainaHussain/Amazon_Review_Analyzer.git

cd sentimart

pip install -r requirements.txt

streamlit run app.py

## 🚧 Active Development & Research Direction

- Sentimart is intentionally designed as an extensible system. Current development and research efforts focus on:
- Mixed and compound emotion detection
- Neutral and ambiguous sentiment classes
- Aspect-based sentiment analysis
- Cross-platform opinion aggregation
- Persistent analytics storage
- Domain-specific fine-tuning
- Model benchmarking and performance evaluation

These additions are not theoretical — the architecture is already structured to support them.
