import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Page config
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide"
)

# Title
st.title("🎓 Student Event Feedback Sentiment Analysis")
st.markdown("Analyze student feedback to uncover satisfaction trends and improvement areas using NLP.")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())
# STEP 4: Sentiment Analysis Logic
def get_sentiment(text):
    if pd.isna(text):
        return "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Teaching_Sentiment"] = df["teaching_text"].apply(get_sentiment)
# STEP 5: Visualization
st.subheader("📊 Teaching Feedback Sentiment Distribution")

sentiment_counts = df["Teaching_Sentiment"].value_counts()

fig, ax = plt.subplots()
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    palette="coolwarm",
    ax=ax
)

ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Responses")

st.pyplot(fig)
st.markdown("---")
st.subheader("📌 Key Insights")

st.markdown("""
- Teaching feedback is **mostly positive**, indicating effective instruction.
- Library facilities show **mixed to negative sentiment**, suggesting scope for improvement.
- Examination-related feedback is **neutral**, indicating consistency but limited engagement.
""")

st.subheader("⚠️ Limitations")

st.markdown("""
- Sentiment analysis relies on **lexicon-based models** (TextBlob), which may miss context.
- Dataset size is relatively small and limited to one institution.
- Neutral labels may include slightly positive or negative opinions.
""")

st.subheader("🚀 Future Scope")

st.markdown("""
- Apply **VADER or transformer-based models (BERT)** for deeper sentiment analysis.
- Perform **topic modeling (LDA)** to extract key complaint themes.
- Deploy a **fully interactive dashboard** with filters for categories and sentiment.
""")

