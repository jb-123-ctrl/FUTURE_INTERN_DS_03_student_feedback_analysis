import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide"
)

# --------------------------------------------------
# LIGHT THEME BACKGROUND
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #F8FAFC, #EAF2FF);
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
st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Event Feedback Sentiment Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#4B5563;'>Advanced NLP-based analysis to uncover satisfaction trends and improvement areas</p>",
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
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🔍 Select Feedback Category")

category_map = {
    "Teaching": "teaching.1",
    "Course Content": "coursecontent.1",
    "Examination": "Examination",
    "Lab Work": "labwork.1",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular.1"
}

selected_category = st.sidebar.selectbox(
    "Category",
    list(category_map.keys())
)

text_column = category_map[selected_category]

# --------------------------------------------------
# SENTIMENT FUNCTION
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

df["Sentiment"] = df[text_column].apply(get_sentiment)

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
sentiment_counts = df["Sentiment"].value_counts()

k1, k2, k3 = st.columns(3)
k1.metric("😊 Positive", sentiment_counts.get("Positive", 0))
k2.metric("😐 Neutral", sentiment_counts.get("Neutral", 0))
k3.metric("😞 Negative", sentiment_counts.get("Negative", 0))

st.markdown("---")

# --------------------------------------------------
# 3D-STYLE PIE CHART (SIMULATED)
# --------------------------------------------------
st.subheader(f"📊 Sentiment Distribution — {selected_category}")

pie_colors = ["#4CC9F0", "#FFD166", "#EF476F"]

pie_fig = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    color_discrete_sequence=pie_colors,
    hole=0.35
)

pie_fig.update_traces(
    pull=[0.06, 0.02, 0.08],  # depth illusion
    rotation=45,
    marker=dict(
        line=dict(color="#FFFFFF", width=3)
    ),
    textinfo="percent+label"
)

pie_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=True
)

st.plotly_chart(pie_fig, use_container_width=True)

# --------------------------------------------------
# SENTIMENT TREND ANALYSIS
# --------------------------------------------------
st.subheader("📈 Sentiment Trend")

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map)
df["Index"] = np.arange(len(df))

trend_fig = px.line(
    df,
    x="Index",
    y="Sentiment_Score",
    markers=True,
    color_discrete_sequence=["#4361EE"]
)

trend_fig.update_layout(
    yaxis=dict(
        tickvals=[-1, 0, 1],
        ticktext=["Negative", "Neutral", "Positive"]
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(trend_fig, use_container_width=True)

# --------------------------------------------------
# WORD CLOUD (LIGHT THEME)
# --------------------------------------------------
st.subheader("☁️ Key Feedback Themes")

combined_text = " ".join(df[text_column].dropna().astype(str))

wordcloud = WordCloud(
    width=900,
    height=400,
    background_color="white",
    colormap="cool",
    max_words=120
).generate(combined_text)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud)
ax.axis("off")

st.pyplot(fig)

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------
with st.expander("📂 Sample Feedback"):
    st.dataframe(df[[text_column, "Sentiment"]].head(10))

# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------
st.subheader("📌 Key Insights")
st.markdown(f"""
- **{selected_category}** feedback is largely **{sentiment_counts.idxmax()}**
- Negative responses indicate actionable improvement areas
- Sentiment trend shows overall consistency
""")

# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------
st.subheader("⚠️ Limitations")
st.markdown("""
- Lexicon-based sentiment may miss sarcasm  
- Dataset limited to a single institution  
- Temporal patterns are not available  
""")

# --------------------------------------------------
# FUTURE SCOPE
# --------------------------------------------------
st.subheader("🚀 Future Enhancements")
st.markdown("""
- BERT-based sentiment classification  
- Topic modeling (LDA) for complaint discovery  
- Real-time feedback dashboard  
- Admin filters & role-based views  
""")

st.markdown(
    "<p style='text-align:center;font-size:12px;color:#6B7280;'>Built with Streamlit & NLP</p>",
    unsafe_allow_html=True
)
