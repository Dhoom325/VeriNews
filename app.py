import streamlit as st
import torch
import requests
import feedparser
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import login
from groq import Groq
 
login(token=st.secrets["HF_TOKEN"])
 
# Config
 
MODEL_PATH = "singhyuvraj999/Roberta_FineTuned"
# MODEL_PATH = "saved_model\\roberta-welfake-final"
MAX_LENGTH = 256
 
SUSPICIOUS_PHRASES = [
    "miracle cure",
    "secret government",
    "what they don't want you to know",
    "hidden truth",
    "shocking discovery",
    "cover up",
    "conspiracy",
    "you won't believe"
]
 
SAFE_ABBREVIATIONS = ["WHO", "CDC", "UN", "EU", "IMF", "NASA"]
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# Load model
 
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model
 
tokenizer, model = load_model()
 
 
# Helpers
 
def combine_title_article(title, article):
    return f"Title: {title} Article: {article}"
 
 
# Keyword extraction
 
def extract_keywords(text, top_n=5):
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        matrix = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        scores = matrix.toarray().flatten()
        ranked = scores.argsort()[::-1]
        keywords = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            keywords.append(features[i])
            if len(keywords) == top_n:
                break
        return keywords
    except:
        return []
 
 
# Fake signal detection
 
def detect_fake_signals(text):
    text_lower = text.lower()
    found = []
    for phrase in SUSPICIOUS_PHRASES:
        if phrase in text_lower:
            found.append(phrase)
    return found
 
 
# Explanation builder
 
def build_fake_reasons(title, article, signals, confidence):
    reasons = []
    keywords = extract_keywords(title + " " + article)
    if keywords:
        reasons.append("Key terms influencing the prediction: " + ", ".join(keywords))
        reasons.append("These terms may contribute to patterns commonly observed in misleading or low-credibility news sources.")
    if signals:
        reasons.append("Sensational or conspiracy-style phrases detected: " + ", ".join(signals))
    text = article or ""
    if text.count("!") >= 3 or text.count("?") >= 3:
        reasons.append("Excessive sensational punctuation detected")
    upper_tokens = [
        tok for tok in text.split()
        if tok.isupper()
        and tok not in SAFE_ABBREVIATIONS
        and len(tok) > 3
    ]
    if len(upper_tokens) >= 4:
        reasons.append("Large number of ALL CAPS words detected")
    if confidence >= 0.85:
        reasons.append("Model confidence for FAKE class is high")
    elif confidence >= 0.65:
        reasons.append("Model confidence for FAKE class is medium")
    else:
        reasons.append("Model confidence is low — verify with trusted sources")
    return reasons
 
 
# Groq LLM Verification
 
def llm_verify(title, article, model_prediction, confidence):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
 
        label = "REAL" if model_prediction == 1 else "FAKE"
        text_snippet = (title + " " + article)[:1000]
 
        prompt = f"""You are a fake news detection expert. A RoBERTa machine learning model predicted this news article is '{label}' with {confidence * 100:.1f}% confidence.
 
Article: {text_snippet}
 
Carefully analyze the writing style, claims, tone, and content. Then respond in this exact format and nothing else:
Verdict: [REAL or FAKE]
Reason: [1-2 sentence explanation]"""
 
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
 
        return response.choices[0].message.content
 
    except Exception as e:
        return f"LLM verification unavailable: {str(e)}"
 
 
# NewsAPI
 
def get_newsapi_key():
    try:
        return st.secrets["NEWS_API_KEY"]
    except:
        return None
 
 
def fetch_newsapi(title, api_key, top_k=3):
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": title,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": top_k,
        "apiKey": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except:
        return []
    articles = []
    for item in data.get("articles", []):
        t = item.get("title")
        source = (item.get("source") or {}).get("name", "Unknown")
        url = item.get("url")
        if t and url:
            articles.append({"title": t, "source": source, "url": url})
    return articles
 
 
# Google RSS
 
def fetch_google_news_rss(title, top_k=3):
    query = title.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:top_k]:
        source = "Google News"
        if hasattr(entry, "source"):
            source = entry.source.title
        articles.append({"title": entry.title, "source": source, "url": entry.link})
    return articles
 
 
# TF-IDF fallback search
 
def tfidf_fallback_search(title, article, top_k=3):
    keywords = extract_keywords(title + " " + article)
    if not keywords:
        return []
    query = " ".join(keywords[:3]).replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:top_k]:
        source = "Google News"
        if hasattr(entry, "source"):
            source = entry.source.title
        articles.append({"title": entry.title, "source": source, "url": entry.link})
    return articles
 
 
# Retrieval pipeline
 
def fetch_verified_news(title, article):
    api_key = get_newsapi_key()
    articles = fetch_newsapi(title, api_key)
    if articles:
        return articles, "api"
    articles = fetch_google_news_rss(title)
    if articles:
        return articles, "rss"
    articles = tfidf_fallback_search(title, article)
    if articles:
        return articles, "tfidf"
    return [], "none"
 
 
# Agent pipeline
 
def ai_agent_pipeline(title, article, prediction, confidence):
    signals = detect_fake_signals(article)
    articles, source_type = fetch_verified_news(title, article)
    fake_reasons = []
    if prediction == 0:
        fake_reasons = build_fake_reasons(title, article, signals, confidence)
    primary = articles[0] if source_type in ["api", "rss"] else None
    return {
        "signals": signals,
        "fake_reasons": fake_reasons,
        "articles": articles,
        "primary": primary,
        "source_type": source_type
    }
 
 
# UI
 
st.title("📰 Fake News Detection")
st.write("Enter a news **title** and **article** to check whether it appears real or fake.")
 
title = st.text_input("News Title")
article = st.text_area("News Article", height=200)
 
 
# Prediction
 
if st.button("Predict"):
 
    if not title.strip() and not article.strip():
        st.warning("Please enter a title or article.")
 
    else:
        text = combine_title_article(title, article)
 
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )
 
        inputs = {k: v.to(device) for k, v in inputs.items()}
 
        with torch.no_grad():
            outputs = model(**inputs)
 
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
 
        if pred == 1:
            st.success("Model Prediction: REAL")
        else:
            st.error("Model Prediction: FAKE")
 
        st.caption(f"Model confidence: {confidence * 100:.1f}%")
 
        # Groq LLM second opinion
        st.markdown("---")
        st.subheader("LLM Verification (Groq | Llama 3)")
 
        with st.spinner("Getting second opinion from LLM..."):
            llm_result = llm_verify(title, article, pred, confidence)
 
        # Parse verdict from LLM response
        if "Verdict: REAL" in llm_result:
            st.success(llm_result)
        elif "Verdict: FAKE" in llm_result:
            st.error(llm_result)
        else:
            st.info(llm_result)
 
        # Agreement/disagreement indicator
        llm_says_real = "Verdict: REAL" in llm_result
        llm_says_fake = "Verdict: FAKE" in llm_result
        model_says_real = pred == 1
 
        if (model_says_real and llm_says_real) or (not model_says_real and llm_says_fake):
            st.success("Both the model and LLM agree on this prediction.")
        elif llm_says_real or llm_says_fake:
            st.warning("The model and LLM disagree — treat this result with extra caution.")
 
        st.markdown("---")
 
        result = ai_agent_pipeline(title, article, pred, confidence)
 
        # FAKE
        if pred == 0:
            st.warning("This article shows patterns sometimes associated with misleading news.")
            st.subheader("Why it was flagged")
            for reason in result["fake_reasons"]:
                st.write("•", reason)
            st.markdown("---")
            st.subheader("Related news coverage")
 
        # REAL
        else:
            if result["primary"]:
                st.success("This article appears consistent with reporting from established news sources.")
                st.subheader("Primary Source")
                st.markdown(f"[{result['primary']['title']}]({result['primary']['url']})")
                st.caption(f"Source: {result['primary']['source']}")
                st.markdown("---")
                st.subheader("Related verified coverage")
            else:
                st.warning("This article shows patterns sometimes associated with misleading news.")
                st.info("The story may be very recent or not indexed by the news APIs yet. Related articles are shown below.")
                st.subheader("Related news coverage")
 
        if not result["articles"]:
            st.info("No related news articles found.")
        else:
            for art in result["articles"]:
                st.markdown(f"[{art['title']}]({art['url']})")
                st.caption(f"Source: {art['source']}")
 
 
# Footer
 
st.markdown("---")
st.caption("Note: For higher accuracy we recommend providing both the news title and the full article.")
