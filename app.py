import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide"
)

# ---------------- DARK THEME ----------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
h1, h2, h3 { color: #EAEAEA; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🎓 Student Event Feedback Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>NLP-powered dashboard to uncover satisfaction trends & improvement areas</p>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Controls")

category_map = {
    "Teaching": "teaching.1",
    "Course Content": "coursecontent.1",
    "Examination": "Examination",
    "Lab Work": "labwork.1",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular.1"
}

category = st.sidebar.selectbox("Select Category", list(category_map.keys()))
text_column = category_map[category]

# ---------------- SENTIMENT FUNCTION ----------------
def get_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df[text_column].apply(get_sentiment)

# ---------------- KPI METRICS ----------------
counts = df["Sentiment"].value_counts()

c1, c2, c3 = st.columns(3)
c1.metric("😊 Positive", counts.get("Positive", 0))
c2.metric("😐 Neutral", counts.get("Neutral", 0))
c3.metric("😞 Negative", counts.get("Negative", 0))

st.markdown("---")

# ---------------- PIE CHART ----------------
st.subheader(f"📊 Sentiment Distribution — {category}")

pie_fig = px.pie(
    values=counts.values,
    names=counts.index,
    color_discrete_sequence=["#00E5FF", "#FFD166", "#EF476F"]
)

st.plotly_chart(pie_fig, use_container_width=True)

# ---------------- TREND ANALYSIS ----------------
st.subheader("📈 Sentiment Trend (Response-wise)")

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map)
df["Response_Index"] = np.arange(len(df))

trend_fig = px.line(
    df,
    x="Response_Index",
    y="Sentiment_Score",
    markers=True,
    labels={"Sentiment_Score": "Sentiment Level"},
    title="Overall Feedback Mood Trend"
)

st.plotly_chart(trend_fig, use_container_width=True)

# ---------------- WORD CLOUD ----------------
st.subheader("☁️ Key Themes from Feedback")

text_data = " ".join(df[text_column].dropna().astype(str))
wordcloud = WordCloud(
    background_color="black",
    colormap="cool",
    width=900,
    height=400
).generate(text_data)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud)
ax.axis("off")
st.pyplot(fig)

# ---------------- SAMPLE DATA ----------------
with st.expander("📂 View Sample Feedback"):
    st.dataframe(df[[text_column, "Sentiment"]].head(10))

# ---------------- INSIGHTS ----------------
st.subheader("📌 Insights")

st.markdown(f"""
- **{category}** feedback is mostly **{counts.idxmax()}**
- Negative responses highlight **specific improvement areas**
- Trend analysis shows overall **sentiment consistency**
""")

# ---------------- LIMITATIONS ----------------
st.subheader("⚠️ Limitations")
st.markdown("""
- Lexicon-based NLP may miss sarcasm & context
- Dataset limited to one institution
- No time-stamp data for temporal analysis
""")

# ---------------- FUTURE SCOPE ----------------
st.subheader("🚀 Future Scope")
st.markdown("""
- BERT / RoBERTa sentiment models
- Topic modeling (LDA)
- Real-time feedback ingestion
- Admin dashboard with filters
""")

st.markdown("<p style='text-align:center;font-size:12px;'>Built with ❤️ using Streamlit & NLP</p>", unsafe_allow_html=True)

