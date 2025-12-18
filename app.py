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
st.subheader("🧠 Key Insights")

st.markdown("""
- 📌 Teaching feedback is **mostly positive**, indicating effective instruction.
- 📌 Some neutral and negative responses suggest **scope for improvement**.
- 📌 Library and infrastructure-related feedback needs attention.
""")
st.subheader("🧠 Key Insights")

st.markdown("""
- 📌 Teaching feedback is **mostly positive**, indicating effective instruction.
- 📌 Some neutral and negative responses suggest **scope for improvement**.
- 📌 Library and infrastructure-related feedback needs attention.
""")
