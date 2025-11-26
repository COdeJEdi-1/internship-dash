import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Link Spread Insights Dashboard", layout="wide")
sns.set(style="whitegrid")

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("processed.csv")
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    return df

df = load_data()

# Helper to pull domain data
def get_domain_data(domain_name):
    return df[df['domain'].str.contains(domain_name, case=False, na=False)]

# Summary extractor
def get_summary(data):
    if data.empty:
        return {
            "Total": 0,
            "Peak": 0,
            "Avg": 0,
            "Top": "None"
        }
    freq = data.resample('D', on='created_utc').size()
    return {
        "Total": len(data),
        "Peak": freq.max(),
        "Avg": round(freq.mean(),2),
        "Top": data['subreddit'].value_counts().idxmax()
    }

# Plot time series
def plot_time_series(data):
    ts = data.resample('D', on='created_utc').size()
    fig, ax = plt.subplots(figsize=(10,4))
    ts.plot(ax=ax)
    ax.set_title("Posts Over Time")
    return fig

# Plot subreddit bars
def plot_subreddit_bar(data):
    counts = data['subreddit'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title("Top Subreddits Sharing This Domain")
    return fig

# ------------------- UI -------------------

st.markdown("<h1>📊 Link Spread Insights Dashboard</h1>", unsafe_allow_html=True)

# Two input columns
col1, col2 = st.columns(2)

domain1 = col1.text_input("Domain 1 (e.g., cnn.com):")
domain2 = col2.text_input("Domain 2 (optional):")

# ------------------- PROCESSING -------------------

if domain1:
    data1 = get_domain_data(domain1)
    summary1 = get_summary(data1)

    st.success(f"{len(data1)} posts found for {domain1}")

    st.subheader("📌 Summary Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Posts", summary1["Total"])
    c2.metric("Peak Posts in a Day", summary1["Peak"])
    c3.metric("Top Subreddit", summary1["Top"])

    st.subheader("📈 Trend Over Time")
    st.pyplot(plot_time_series(data1))

    st.subheader("🔥 Top Subreddits")
    st.pyplot(plot_subreddit_bar(data1))


# ------------------- DOMAIN COMPARISON -------------------

if domain1 and domain2:
    data2 = get_domain_data(domain2)
    summary2 = get_summary(data2)

    st.markdown("---")
    st.subheader("⚔️ Domain Comparison")

    comparison_df = pd.DataFrame({
        "Metric": ["Total Posts", "Peak Posts", "Avg Posts/Day", "Top Subreddit"],
        domain1: [summary1["Total"], summary1["Peak"], summary1["Avg"], summary1["Top"]],
        domain2: [summary2["Total"], summary2["Peak"], summary2["Avg"], summary2["Top"]],
    })

    st.table(comparison_df)

    # Side-by-side charts
    st.markdown("### 📊 Activity Comparison Chart")

    fig, ax = plt.subplots(figsize=(10,4))
    data1.resample('D', on='created_utc').size().plot(ax=ax, label=domain1)
    data2.resample('D', on='created_utc').size().plot(ax=ax, label=domain2)
    ax.legend()
    ax.set_title("Posting Activity Comparison")
    st.pyplot(fig)

