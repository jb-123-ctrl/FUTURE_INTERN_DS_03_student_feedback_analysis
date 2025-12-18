import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Event Feedback Sentiment Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Advanced NLP-based analysis of student feedback to identify satisfaction trends and improvement areas.</p>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# ---------------- CATEGORY SELECTION ----------------
category_map = {
    "Teaching": "teaching_text",
    "Course Content": "coursecontent_text",
    "Examination": "examination_text",
    "Lab Work": "labwork_text",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular_text"
}

st.sidebar.header("⚙️ Controls")
selected_category = st.sidebar.selectbox(
    "Select Feedback Category",
    list(category_map.keys())
)

text_column = category_map[selected_category]

# ---------------- SENTIMENT FUNCTION ----------------
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

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

sentiment_counts = df["Sentiment"].value_counts()

col1.metric("😊 Positive", sentiment_counts.get("Positive", 0))
col2.metric("😐 Neutral", sentiment_counts.get("Neutral", 0))
col3.metric("😞 Negative", sentiment_counts.get("Negative", 0))

st.markdown("---")

# ---------------- PIE CHART ----------------
st.subheader(f"📊 Sentiment Distribution — {selected_category}")

fig_pie = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    hole=0.45,
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- WORD CLOUD ----------------
st.subheader("☁️ Word Cloud (Most Frequent Terms)")

text_data = " ".join(df[text_column].dropna().astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate(text_data)

fig_wc, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)

# ---------------- SAMPLE DATA ----------------
with st.expander("📂 View Sample Feedback"):
    st.dataframe(df[[text_column, "Sentiment"]].head(10))

# ---------------- INSIGHTS ----------------
st.markdown("---")
st.subheader("📌 Key Insights")

st.markdown(f"""
- **{selected_category} feedback** shows a dominant **{sentiment_counts.idxmax()} sentiment**.
- Negative responses highlight areas requiring targeted improvements.
- Positive sentiment indicates strengths that should be retained and scaled.
""")

# ---------------- LIMITATIONS ----------------
st.subheader("⚠️ Limitations")

st.markdown("""
- Lexicon-based sentiment analysis may miss sarcasm and context.
- Dataset is limited to a single institution.
- Neutral sentiment may include mixed opinions.
""")

# ---------------- FUTURE SCOPE ----------------
st.subheader("🚀 Future Scope")

st.markdown("""
- Integrate **VADER / BERT** for deeper sentiment understanding.
- Apply **topic modeling (LDA)** to extract complaint themes.
- Add **time-based trend analysis**.
- Deploy feedback collection with **real-time dashboards**.
""")

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Built with ❤️ using Streamlit & NLP</p>",
    unsafe_allow_html=True
)
