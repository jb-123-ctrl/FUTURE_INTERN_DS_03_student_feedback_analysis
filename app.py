import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    page_icon="🎓",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🎓 Student Event Feedback Sentiment Analysis")
st.markdown(
    "Analyze student feedback to uncover satisfaction trends and improvement areas using NLP."
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# ---------------- SENTIMENT FUNCTION ----------------
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

# ---------------- CATEGORY SELECTOR ----------------
category_map = {
    "Teaching": "teaching_text",
    "Course Content": "coursecontent_text",
    "Examination": "examination_text",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular_text"
}

selected_category = st.selectbox(
    "📌 Select Feedback Category",
    list(category_map.keys())
)

text_column = category_map[selected_category]

df["Sentiment"] = df[text_column].apply(get_sentiment)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🧠 Sentiment Analysis",
    "📌 Insights & Scope"
])

# ================= TAB 1 =================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📂 Dataset Preview")
        st.dataframe(df[[text_column]].head())

    with col2:
        st.subheader("📊 Dataset Summary")
        st.metric("Total Responses", len(df))
        st.metric("Selected Category", selected_category)

# ================= TAB 2 =================
with tab2:
    col1, col2 = st.columns(2)

    # -------- PIE CHART --------
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title=f"{selected_category} Feedback Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#4CC9F0",
            "Negative": "#F72585",
            "Neutral": "#7209B7"
        },
        hole=0.4
    )

    fig.update_layout(height=380)

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # -------- WORD CLOUD --------
    with col2:
        st.subheader("☁️ Feedback Word Cloud")

        text_data = " ".join(
            df[text_column].dropna().astype(str).tolist()
        )

        wordcloud = WordCloud(
            width=600,
            height=350,
            background_color="white",
            colormap="viridis"
        ).generate(text_data)

        fig_wc, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig_wc)

# ================= TAB 3 =================
with tab3:
    st.subheader("📌 Key Insights")

    st.markdown(f"""
    - **{selected_category} feedback** is predominantly **positive**, indicating overall satisfaction.
    - **Negative responses** highlight specific improvement areas.
    - **Neutral feedback** suggests consistency but room for engagement.
    """)

    st.subheader("⚠️ Limitations")
    st.markdown("""
    - Sentiment analysis uses **TextBlob (lexicon-based)** and may miss sarcasm or context.
    - Dataset is limited to a **single institution**.
    - Neutral sentiment may include weak opinions.
    """)

    st.subheader("🚀 Future Scope")
    st.markdown("""
    - Use **VADER / BERT-based models** for deeper sentiment understanding.
    - Perform **topic modeling (LDA)** to identify key complaint themes.
    - Deploy as a **real-time feedback dashboard** integrated with Google Forms.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("📊 Built with Streamlit | NLP | Data Science Internship Project")

