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
