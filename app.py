import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide"
)

# --------------------------------------------------
# LIGHT SEA-BLUE + GREEN THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E0F7FA, #E8F5E9);
    color: #1F2933;
}
h1, h2, h3 {
    color: #1F2933;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🎓 Student Event Feedback Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#374151;'>Advanced NLP dashboard to uncover satisfaction trends & improvement areas</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# --------------------------------------------------
# SIDEBAR FILTER
# --------------------------------------------------
st.sidebar.header("🔍 Controls")

category_map = {
    "Teaching": "teaching.1",
    "Course Content": "coursecontent.1",
    "Examination": "Examination",
    "Lab Work": "labwork.1",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular.1"
}

category = st.sidebar.selectbox("Select Feedback Category", list(category_map.keys()))
text_column = category_map[category]

# --------------------------------------------------
# SENTIMENT FUNCTIONS
# --------------------------------------------------
def get_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def get_polarity(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

df["Sentiment"] = df[text_column].apply(get_sentiment)
df["Polarity"] = df[text_column].apply(get_polarity)

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
total = len(df)
positive_pct = round((df["Sentiment"] == "Positive").sum() / total * 100, 1)
negative_pct = round((df["Sentiment"] == "Negative").sum() / total * 100, 1)
risk_count = (df["Sentiment"] == "Negative").sum()

k1, k2, k3 = st.columns(3)
k1.metric("😊 Satisfaction Score", f"{positive_pct}%")
k2.metric("⚠️ Risk Area Count", risk_count)
k3.metric("📉 Negative Trend %", f"{negative_pct}%")

st.markdown("---")

# --------------------------------------------------
# STAKEHOLDER TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎓 Admin View", "👨‍🏫 Faculty View", "🧑‍🎓 Student View"])

# ================= ADMIN VIEW =================
with tab1:
    st.subheader(f"📊 Sentiment Distribution — {category}")

    counts = df["Sentiment"].value_counts()

    pie_fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.35,
        color_discrete_sequence=["#2EC4B6", "#90DBF4", "#EF476F"]
    )

    pie_fig.update_traces(
        pull=[0.05, 0.02, 0.07],
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=2))
    )

    pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(pie_fig, use_container_width=True)

# ================= FACULTY VIEW =================
with tab2:
    st.subheader("📈 Sentiment Trend")

    df["Index"] = np.arange(len(df))
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map)

    trend_fig = px.line(
        df,
        x="Index",
        y="Sentiment_Score",
        markers=True,
        color_discrete_sequence=["#2EC4B6"]
    )

    trend_fig.update_layout(
        yaxis=dict(
            tickvals=[-1, 0, 1],
            ticktext=["Negative", "Neutral", "Positive"]
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(trend_fig, use_container_width=True)

# ================= STUDENT VIEW =================
with tab3:
    st.subheader("☁️ Feedback Themes (WordCloud)")

    raw_text = df[text_column].dropna().astype(str)

    cleaned_text = []
    for text in raw_text:
        text = re.sub(r"[^a-zA-Z ]", "", text.lower()).strip()
        if len(text.split()) > 2:
            cleaned_text.append(text)

    final_text = " ".join(cleaned_text)

    if not final_text or len(final_text.split()) < 5:
        st.info("Not enough meaningful feedback to generate WordCloud for this category.")
    else:
        wordcloud = WordCloud(
            width=800,
            height=350,
            background_color="white",
            colormap="summer",
            max_words=120,
            stopwords=set(WordCloud().stopwords)
        ).generate(final_text)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

# --------------------------------------------------
# KEYWORD FREQUENCY ANALYSIS
# --------------------------------------------------
st.markdown("---")
st.subheader("🔑 Keyword Frequency Analysis")

def extract_keywords(text_series):
    words = []
    for text in text_series.dropna():
        clean = re.sub(r"[^a-zA-Z ]", "", text.lower())
        words.extend(clean.split())
    return Counter(words)

pos_words = extract_keywords(df[df["Sentiment"] == "Positive"][text_column])
neg_words = extract_keywords(df[df["Sentiment"] == "Negative"][text_column])

c1, c2 = st.columns(2)

with c1:
    st.markdown("### ✅ Positive Keywords")
    for w, c in pos_words.most_common(8):
        st.write(f"• **{w}** ({c})")

with c2:
    st.markdown("### ❌ Negative Keywords")
    for w, c in neg_words.most_common(8):
        st.write(f"• **{w}** ({c})")

# --------------------------------------------------
# SENTIMENT HEATMAP
# --------------------------------------------------
st.markdown("---")
st.subheader("🔥 Sentiment Heatmap Across Categories")

heatmap_data = {}
for cat, col in category_map.items():
    heatmap_data[cat] = df[col].dropna().apply(get_sentiment).value_counts()

heatmap_df = pd.DataFrame(heatmap_data).fillna(0)

fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    ax=ax
)
ax.set_xlabel("Category")
ax.set_ylabel("Sentiment")
st.pyplot(fig)

# --------------------------------------------------
# POLARITY DISTRIBUTION
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 Sentiment Intensity Distribution")

hist_fig = px.histogram(
    df,
    x="Polarity",
    nbins=30,
    color_discrete_sequence=["#2EC4B6"]
)

hist_fig.update_layout(
    xaxis_title="Polarity (-1 = Negative, +1 = Positive)",
    yaxis_title="Responses",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(hist_fig, use_container_width=True)

# --------------------------------------------------
# TOPIC MODELING (LDA)
# --------------------------------------------------
st.markdown("---")
st.subheader("🧠 Topic Modeling (LDA)")

try:
    vectorizer = CountVectorizer(stop_words="english", max_df=0.9, min_df=5)
    dtm = vectorizer.fit_transform(df[text_column].dropna().astype(str))

    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        words = [feature_names[j] for j in topic.argsort()[:-7:-1]]
        st.markdown(f"**Topic {i+1}:** {', '.join(words)}")

except:
    st.warning("Topic modeling could not be generated due to limited data.")

# --------------------------------------------------
# LIMITATIONS & FUTURE
# --------------------------------------------------
st.subheader("⚠️ Limitations")
st.markdown("""
- Lexicon-based sentiment may miss sarcasm or context  
- Dataset limited to one institution  
- Topic modeling depends on text volume  
""")

st.subheader("🚀 Future Scope")
st.markdown("""
- BERT-based sentiment analysis  
- Automated recommendation engine  
- Department-level dashboards  
- Real-time feedback ingestion  
""")

st.markdown(
    "<p style='text-align:center;font-size:12px;color:#374151;'>Built with Streamlit & Advanced NLP</p>",
    unsafe_allow_html=True
)
