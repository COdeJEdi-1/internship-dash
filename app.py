import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvis.network import Network
import streamlit.components.v1 as components


# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("processed.csv")
df['created_utc'] = pd.to_datetime(df['created_utc'])


# -------------------------------
# Initialize NLP Models
# -------------------------------
@st.cache_resource
def load_models():
    sentiment_model = SentimentIntensityAnalyzer()
    toxicity_pipe = pipeline("text-classification", model="unitary/toxic-bert", truncation=True)
    emotion_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", top_k=1)
    return sentiment_model, toxicity_pipe, emotion_pipe

sentiment_model, toxicity_pipe, emotion_pipe = load_models()


# -------------------------------
# Apply NLP Scoring
# -------------------------------
@st.cache_data
def compute_text_features(data):
    data = data.copy()

    # sentiment
    data["sentiment"] = data["title"].apply(lambda x: sentiment_model.polarity_scores(str(x))['compound'])

    # toxicity
    data["toxicity"] = data["title"].apply(lambda x: toxicity_pipe(str(x))[0]['score'])

    # emotion
    def safe_emotion(text):
        try:
            return emotion_pipe(str(text))[0][0]['label']
        except:
            return "unknown"

    data["emotion"] = data["title"].apply(safe_emotion)

    return data


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Link Spread Intelligence Dashboard")

domain1 = st.text_input("Enter a news domain (e.g., cnn.com):")
domain2 = st.text_input("Optional second domain for comparison:")

if domain1:
    filtered = df[df['domain'].str.contains(domain1, case=False, na=False)]
    filtered = compute_text_features(filtered)

    st.success(f"{len(filtered)} posts found for {domain1}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(filtered))
    col2.metric("Avg Sentiment", round(filtered['sentiment'].mean(), 3))
    col3.metric("Avg Toxicity", round(filtered['toxicity'].mean(), 3))

    # Emotion chart
    st.subheader("🧠 Emotion Distribution")
    emotion_counts = filtered['emotion'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=emotion_counts.values, y=emotion_counts.index, ax=ax)
    st.pyplot(fig)

    # Time series trend
    st.subheader("📈 Trend Over Time")
    ts = filtered.resample('D', on='created_utc').size()
    fig2, ax2 = plt.subplots()
    ts.plot(ax=ax2)
    st.pyplot(fig2)

    # Top Subreddits
    st.subheader("🔥 Top Subreddits Posting This Domain")
    sr = filtered['subreddit'].value_counts().head(10)
    fig3, ax3 = plt.subplots()
    sns.barplot(x=sr.values, y=sr.index, ax=ax3)
    st.pyplot(fig3)


    # -------------------------------
    # Network Graph
    # -------------------------------
    st.subheader("🕸 Network Graph: How This Domain Spreads Across Reddit")

    def create_network_graph(df, domain):
        net = Network(height="550px", width="100%", bgcolor="#111", font_color="white")

        net.add_node(domain, label=domain, color="#00c3ff", size=25)

        subreddit_counts = df['subreddit'].value_counts().head(15)
        for subreddit, count in subreddit_counts.items():
            net.add_node(subreddit, label=subreddit, size=15 + (count / 3), color="#ff6b6b")
            net.add_edge(domain, subreddit, value=count)

        net.force_atlas_2based()
        return net

    if not filtered.empty:
        net = create_network_graph(filtered, domain1)

        path = "network_graph.html"
        net.save_graph(path)

        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        components.html(html, height=600, scrolling=False)
    else:
        st.info("Not enough data to build a graph.")


    # -------------------------------
    # Topic Clustering
    # -------------------------------
    st.subheader("🧩 Topic Clustering (Auto-Generated Themes)")

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(filtered['title'])
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered['cluster'] = kmeans.fit_predict(X)

    st.write(filtered[['title', 'cluster']].head(20))


    # -------------------------------
    # Narrative Summary
    # -------------------------------
    st.subheader("📌 AI Summary Report")

    def generate_story(data, dom):
        tone = "negative" if data['sentiment'].mean() < -0.1 else "mixed" if data['sentiment'].mean() < 0.1 else "positive"
        top_emotion = data['emotion'].value_counts().idxmax()
        top_sub = data['subreddit'].value_counts().idxmax()

        return f"""
        The domain **{dom}** appears **{len(data)} times** across Reddit.
        The overall conversational tone is **{tone}**, dominated emotionally by **{top_emotion}**.
        It gains most traction in **r/{top_sub}**.

        Posting activity peaked at **{data.resample('D', on='created_utc').size().max()} posts/day**, suggesting an event-driven influence.

        The clustering model identified **{data['cluster'].nunique()} distinct topic groups**, indicating diverse discussions around this source.
        """

    st.info(generate_story(filtered, domain1))


# -------------------------------
# Comparison: Domain1 vs Domain2
# -------------------------------
if domain1 and domain2:
    st.subheader("⚔️ Activity Comparison")

    def get_ts(dom):
        d = compute_text_features(df[df['domain'].str.contains(dom, case=False, na=False)])
        return d.resample('D', on='created_utc').size()

    ts1, ts2 = get_ts(domain1), get_ts(domain2)

    fig4, ax4 = plt.subplots()
    ts1.plot(ax=ax4, label=domain1)
    ts2.plot(ax=ax4, label=domain2)
    ax4.legend()
    st.pyplot(fig4)
