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
# SEA BLUE + GREEN BACKGROUND THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E0F7FA, #E8F5E9);
    color: #1F2933;
}
h1, h2, h3 {
    color: #1F2933;
}
.sidebar .sidebar-content {
    background-color: #E0F2F1;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Feedback Sentiment Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#374151;'>Advanced NLP dashboard to uncover satisfaction trends & improvement areas</p>",
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
# STAKEHOLDER TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎓 Admin View",
    "👨‍🏫 Faculty View",
    "🧑‍🎓 Student View"
])

# ================= ADMIN VIEW =================
with tab1:
    st.subheader(f"📊 Sentiment Distribution — {selected_category}")

    pie_fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color_discrete_sequence=["#2EC4B6", "#90DBF4", "#EF476F"],
        hole=0.35
    )

    pie_fig.update_traces(
        pull=[0.06, 0.03, 0.08],   # 3D-style depth illusion
        rotation=40,
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=3))
    )

    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("""
    **Admin Insight:**  
    This overview helps management identify satisfaction levels and risk areas quickly.
    """)

# ================= FACULTY VIEW =================
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
        color_discrete_sequence=["#2EC4B6"]
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
    Trend patterns help instructors understand consistency and emotional response.
    """)

# ================= STUDENT VIEW =================
with tab3:
    st.subheader("☁️ Key Feedback Themes")

    combined_text = " ".join(df[text_column].dropna().astype(str))

    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color="white",
        colormap="summer",
        max_words=120
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

    st.markdown("""
    **Student Insight:**  
    Frequently mentioned terms highlight common experiences and expectations.
    """)

# --------------------------------------------------
# TOPIC MODELING (LDA)
# --------------------------------------------------
st.markdown("---")
st.subheader("🧠 Topic Modeling (LDA) — Hidden Themes")

text_data = df[text_column].dropna().astype(str)

try:
    vectorizer = CountVectorizer(stop_words="english", max_df=0.9, min_df=5)
    dtm = vectorizer.fit_transform(text_data)

    lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
    lda_model.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    for i, topic in enumerate(lda_model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-7:-1]]
        st.markdown(f"**Topic {i+1}:** {', '.join(top_words)}")

except:
    st.warning("Topic modeling could not be generated due to limited data.")

# --------------------------------------------------
# LIMITATIONS & FUTURE SCOPE
# --------------------------------------------------
st.subheader("⚠️ Limitations")
st.markdown("""
- Lexicon-based sentiment analysis may miss sarcasm  
- Dataset limited to one institution  
- Topic modeling depends on text volume  
""")

st.subheader("🚀 Future Enhancements")
st.markdown("""
- Transformer-based models (BERT)  
- Time-based sentiment trends  
- Real-time feedback dashboards  
- Role-based analytics views  
""")

st.markdown(
    "<p style='text-align:center;font-size:12px;color:#374151;'>Built with Streamlit & NLP</p>",
    unsafe_allow_html=True
)


