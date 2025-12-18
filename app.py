import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
    "<p style='text-align:center;color:#4B5563;'>Advanced NLP dashboard with sentiment analysis & topic modeling</p>",
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
# SIDEBAR CONTROLS
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
# STAKEHOLDER TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎓 Admin View",
    "👨‍🏫 Faculty View",
    "🧑‍🎓 Student View"
])

# ==================================================
# 🎓 ADMIN VIEW
# ==================================================
with tab1:
    st.subheader("📊 Overall Sentiment Distribution")

    pie_fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        hole=0.35,
        color_discrete_sequence=["#4CC9F0", "#FFD166", "#EF476F"]
    )

    pie_fig.update_traces(
        pull=[0.06, 0.02, 0.08],
        rotation=40,
        textinfo="percent+label"
    )

    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("""
    **Admin Insight:**  
    This view highlights high-level satisfaction trends to support strategic decision-making.
    """)

# ==================================================
# 👨‍🏫 FACULTY VIEW
# ==================================================
with tab2:
    st.subheader("📈 Sentiment Trend Across Responses")

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

    st.markdown("""
    **Faculty Insight:**  
    Trends help educators understand consistency and emotional response patterns.
    """)

# ==================================================
# 🧑‍🎓 STUDENT VIEW
# ==================================================
with tab3:
    st.subheader("☁️ Key Feedback Themes (Word Cloud)")

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

    st.markdown("""
    **Student Insight:**  
    Frequently mentioned terms highlight what students care about most.
    """)

# --------------------------------------------------
# TOPIC MODELING (LDA)
# --------------------------------------------------
st.markdown("---")
st.subheader("🧠 Topic Modeling (LDA) — Hidden Themes")

text_data = df[text_column].dropna().astype(str)

vectorizer = CountVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=5
)

dtm = vectorizer.fit_transform(text_data)

lda_model = LatentDirichletAllocation(
    n_components=3,
    random_state=42
)

lda_model.fit(dtm)

feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, num_words=6):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(", ".join(top_words))
    return topics

topics = display_topics(lda_model, feature_names)

for i, topic in enumerate(topics, start=1):
    st.markdown(f"**Topic {i}:** {topic}")

# --------------------------------------------------
# LIMITATIONS & FUTURE
# --------------------------------------------------
st.subheader("⚠️ Limitations")
st.markdown("""
- Topic modeling depends on dataset size and text quality  
- Lexicon-based sentiment may miss sarcasm  
- Feedback lacks time-based tracking  
""")

st.subheader("🚀 Future Scope")
st.markdown("""
- Transformer-based sentiment models (BERT)  
- Topic trend evolution over time  
- Real-time feedback ingestion  
- Department-level dashboards  
""")

st.markdown(
    "<p style='text-align:center;font-size:12px;color:#6B7280;'>Advanced NLP Dashboard | Streamlit</p>",
    unsafe_allow_html=True
)
