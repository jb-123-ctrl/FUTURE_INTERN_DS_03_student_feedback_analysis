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
# CUSTOM GRADIENT BACKGROUND
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    color: #F5F7FA;
}
h1, h2, h3 {
    color: #F5F7FA;
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
    "<p style='text-align:center;'>Advanced NLP-driven insights for student satisfaction & improvement</p>",
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
st.sidebar.header("🔍 Analysis Controls")

category_map = {
    "Teaching": "teaching.1",
    "Course Content": "coursecontent.1",
    "Examination": "Examination",
    "Lab Work": "labwork.1",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular.1"
}

selected_category = st.sidebar.selectbox(
    "Select Feedback Category",
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

c1, c2, c3 = st.columns(3)
c1.metric("😊 Positive", sentiment_counts.get("Positive", 0))
c2.metric("😐 Neutral", sentiment_counts.get("Neutral", 0))
c3.metric("😞 Negative", sentiment_counts.get("Negative", 0))

st.markdown("---")

# --------------------------------------------------
# 3D-STYLE PIE CHART (SIMULATED DEPTH)
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
    pull=[0.05, 0.02, 0.08],  # depth illusion
    textinfo="percent+label",
    marker=dict(line=dict(color="#0B1C2D", width=2))
)

pie_fig.update_layout(
    showlegend=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(pie_fig, use_container_width=True)

# --------------------------------------------------
# SENTIMENT TREND ANALYSIS
# --------------------------------------------------
st.subheader("📈 Sentiment Trend Across Responses")

sentiment_score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
df["Sentiment_Score"] = df["Sentiment"].map(sentiment_score_map)
df["Index"] = np.arange(len(df))

trend_fig = px.line(
    df,
    x="Index",
    y="Sentiment_Score",
    markers=True,
    color_discrete_sequence=["#4CC9F0"]
)

trend_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(tickvals=[-1, 0, 1], ticktext=["Negative", "Neutral", "Positive"])
)

st.plotly_chart(trend_fig, use_container_width=True)

# --------------------------------------------------
# WORD CLOUD (UNIQUE BACKGROUND)
# --------------------------------------------------
st.subheader("☁️ Dominant Feedback Themes")

combined_text = " ".join(df[text_column].dropna().astype(str))

wordcloud = WordCloud(
    width=900,
    height=400,
    background_color=None,
    mode="RGBA",
    colormap="winter",
    max_words=120
).generate(combined_text)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud)
ax.axis("off")
fig.patch.set_alpha(0)

st.pyplot(fig)

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------
with st.expander("📂 View Sample Feedback"):
    st.dataframe(df[[text_column, "Sentiment"]].head(10))

# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------
st.subheader("📌 Key Insights")
st.markdown(f"""
- **{selected_category}** feedback is predominantly **{sentiment_counts.idxmax()}**
- Negative sentiment highlights targeted improvement areas
- Trend analysis shows overall sentiment stability
""")

# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------
st.subheader("⚠️ Limitations")
st.markdown("""
- TextBlob may miss sarcasm or contextual nuances  
- Dataset is limited to a single institution  
- Lack of time-based feedback restricts temporal analysis  
""")

# --------------------------------------------------
# FUTURE SCOPE
# --------------------------------------------------
st.subheader("🚀 Future Enhancements")
st.markdown("""
- Integrate **BERT-based sentiment models**  
- Apply **topic modeling (LDA)** for complaint clustering  
- Enable real-time feedback ingestion  
- Add admin-level filters and dashboards  
""")

st.markdown(
    "<p style='text-align:center;font-size:12px;'>Built with ❤️ using Streamlit & NLP</p>",
    unsafe_allow_html=True
)
