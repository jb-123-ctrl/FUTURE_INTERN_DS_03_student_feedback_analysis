import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Student Event Feedback Sentiment Analysis",
    layout="wide"
)

# -------------------- TITLE --------------------
st.title("🎓 Student Event Feedback Sentiment Analysis")
st.markdown(
    "Analyze student feedback to uncover satisfaction trends and improvement areas using NLP."
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# -------------------- RENAME COLUMNS (IMPORTANT FIX) --------------------
df = df.rename(columns={
    "teaching.1": "teaching_text",
    "coursecontent.1": "coursecontent_text",
    "Examination": "examination_text",
    "labwork.1": "labwork_text",
    "library_facilities": "library_text",
    "extracurricular.1": "extracurricular_text"
})

# -------------------- DATA PREVIEW --------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -------------------- SENTIMENT FUNCTION --------------------
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

# -------------------- SENTIMENT ANALYSIS --------------------
df["Teaching_Sentiment"] = df["teaching_text"].apply(get_sentiment)

# -------------------- SENTIMENT DISTRIBUTION CHART --------------------
st.subheader("📊 Teaching Feedback Sentiment Distribution")

sentiment_counts = df["Teaching_Sentiment"].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    palette="coolwarm",
    ax=ax
)

ax.set_title("Teaching Feedback Sentiment Distribution", fontsize=14)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Responses")

# Show values on bars
for i, v in enumerate(sentiment_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center')

st.pyplot(fig)

# -------------------- INSIGHTS --------------------
st.markdown("---")
st.subheader("📌 Key Insights")

st.markdown("""
- Teaching feedback is **mostly positive**, indicating effective instruction quality.
- Some responses highlight the need for **more practical exposure** in teaching.
- Neutral feedback suggests consistency but limited emotional engagement.
""")

# -------------------- LIMITATIONS --------------------
st.subheader("⚠️ Limitations")

st.markdown("""
- Sentiment analysis is based on **lexicon-based NLP (TextBlob)** and may miss sarcasm or context.
- Dataset size is relatively small and limited to a **single institution**.
- Neutral sentiment may include slightly positive or negative opinions.
""")

# -------------------- FUTURE SCOPE --------------------
st.subheader("🚀 Future Scope")

st.markdown("""
- Integrate **VADER or transformer-based models (BERT)** for more accurate sentiment analysis.
- Apply **topic modeling (LDA)** to identify recurring feedback themes.
- Extend the dashboard with **filters by category and sentiment type**.
- Deploy the application with **Streamlit Cloud** for public access.
""")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("📊 Built with Streamlit | NLP using TextBlob | Internship Task 03")
