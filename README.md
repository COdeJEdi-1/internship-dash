# Link Spread Intelligence Dashboard

This project is my assignment submission for the SimPPL internship.

It is an investigative dashboard that analyzes how external links (domains) spread across Reddit, focusing on:

- Which communities amplify a given link  
- How discussion volume evolves over time  
- What tone (sentiment, emotion, toxicity) accompanies the posts  
- What topics or narratives they cluster into  

The dashboard is built using **Streamlit** and is publicly hosted.

---

## 1. Dataset

The dashboard operates on a preprocessed Reddit dataset (`processed.csv`) originating from a JSONL export.  
Each row represents a Reddit post and contains key fields such as:

| Column | Description | Example |
|--------|-------------|---------|
| `domain` | External link domain | `cnn.com`, `nypost.com`, `youtube.com` |
| `subreddit` | Community where link was posted | `politics`, `Conservative` |
| `created_utc` | Timestamp of post | `2024-05-12 16:22:01` |
| `title` | Reddit post headline | `"Biden announces AI policy changes"` |

The focus is on **link-level propagation**, not full comment threads.

---

## 2. Dashboard Features

### 2.1 Summary Insights
After entering a domain, the dashboard computes and displays:

- Total number of posts
- Average sentiment score (VADER)
- Average toxicity (transformer-based toxicity classifier)

Users can optionally compare these metrics between two domains.

---

### 2.2 Time-Series Trends
A visual timeline shows how frequently the selected domain appears over time.

If two domains are provided, a combined activity comparison chart illustrates differences in attention peaks—fulfilling the rubric requirement for time-series visualization.

---

### 2.3 Community Amplifiers
A bar chart displays the **top subreddits posting the domain**, helping identify which communities contribute most to its spread.

---

### 2.4 Tone & Emotion Analysis

For every post title, the dashboard computes:

| Metric | Model |
|--------|-------|
| Sentiment | `vaderSentiment` |
| Toxicity | `unitary/toxic-bert` |
| Emotion | `cardiffnlp/twitter-roberta-base-emotion` |

Results are visualized as:

- Emotion distribution plot  
- Summary metrics (sentiment, toxicity)

---

### 2.5 Topic/Narrative Clustering

To understand narratives, the dashboard performs lightweight topic modeling:

- TF-IDF vectorization
- KMeans clustering  
- Display of sample posts per cluster  

This allows users to inspect the dominant themes associated with the domain.

---

### 2.6 AI Narrative Summary

For each domain, the dashboard automatically generates a written insight summarizing:

- Discussion frequency
- Sentiment and emotional tone  
- Subreddits driving amplification  
- Activity peaks
- Number of distinct narrative clusters  

This supports non-technical interpretation and aligns with the rubric’s "GenAI-based summarization" recommendation.

---

## 3. How to Use

1. Open the hosted dashboard.
2. Input a domain such as:
   - `cnn.com`
   - `nypost.com`
   - `youtube.com`
3. Optionally select a second domain for comparison.
4. Scroll through the dashboard to explore:
   - Summary metrics  
   - Activity over time  
   - Emotion and toxicity maps  
   - Top subreddits  
   - Topic clusters  
   - Generated narrative summary  

Users may adjust domains interactively to run exploratory investigations.

---

## 4. Technical Stack

| Component | Technology |
|-----------|------------|
| Interface | Streamlit |
| Data Processing | pandas |
| Visualizations | matplotlib, seaborn |
| ML/NLP Models | VADER, Toxic-BERT, Twitter-RoBERTa emotion classifier |
| Topic Modeling | TF-IDF + KMeans (scikit-learn) |

---

