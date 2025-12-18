import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide"
)

# =============================
# LIGHT THEME (SEA BLUE + GREEN)
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E8F8F5, #E3F2FD);
}
.section {
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 12px;
}
.good { background-color: #D1FAE5; }
.warn { background-color: #FEF3C7; }
.bad { background-color: #FEE2E2; }
.small-text { font-size: 13px; color: #374151; }
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown(
    "<h2 style='text-align:center;'>🎓 Student Event Feedback Sentiment Analysis</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;' class='small-text'>Compact NLP Dashboard for Stakeholders</p>",
    unsafe_allow_html=True
)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# =============================
# SIDEBAR FILTER
# =============================
st.sidebar.header("🔍 Filters")

category_map = {
    "Teaching": "teaching.1",
    "Course Content": "coursecontent.1",
    "Examination": "Examination",
    "Lab Work": "labwork.1",
    "Library Facilities": "library_facilities",
    "Extracurricular": "extracurricular.1"
}

category = st.sidebar.selectbox("Select Feedback Area", list(category_map.keys()))
text_col = category_map[category]

# =============================
# SENTIMENT FUNCTIONS
# =============================
def polarity(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

df["Polarity"] = df[text_col].apply(polarity)

df["Sentiment"] = df["Polarity"].apply(
    lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
)

# =============================
# KPI METRICS
# =============================
pos_pct = round((df["Polarity"] > 0).mean() * 100, 1)
neg_pct = round((df["Polarity"] < 0).mean() * 100, 1)
risk = (df["Polarity"] < -0.3).sum()

c1, c2, c3 = st.columns(3)
c1.metric("😊 Satisfaction %", f"{pos_pct}%")
c2.metric("⚠️ Risk Responses", risk)
c3.metric("📉 Negative %", f"{neg_pct}%")

# =============================
# TABS FOR STAKEHOLDERS
# =============================
admin_tab, faculty_tab, student_tab = st.tabs(
    ["🎓 Admin View", "👨‍🏫 Faculty View", "🧑‍🎓 Student View"]
)

# =================================================
# ADMIN VIEW
# =================================================
with admin_tab:
    st.markdown("### 📊 Overall Sentiment Distribution")

    pie = px.pie(
        df,
        names="Sentiment",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    pie.update_traces(pull=[0.05, 0.05, 0.05])
    st.plotly_chart(pie, use_container_width=True)

    st.markdown("### 🔥 Sentiment Heatmap")

    heat = df["Sentiment"].value_counts().reset_index()
    heat.columns = ["Sentiment", "Count"]

    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(
        heat[["Count"]].T,
        annot=True,
        cmap="YlGnBu",
        fmt="d",
        ax=ax
    )
    st.pyplot(fig)

# =================================================
# FACULTY VIEW
# =================================================
with faculty_tab:
    st.markdown("### ☁️ Key Feedback Themes")

    text_data = " ".join(df[text_col].dropna().astype(str))
    wc = WordCloud(
        width=600,
        height=250,
        background_color="white",
        colormap="summer"
    ).generate(text_data)

    fig_wc, ax_wc = plt.subplots(figsize=(6, 2.5))
    ax_wc.imshow(wc)
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.markdown("### 🧠 Topic Modeling (LDA)")

    try:
        vec = CountVectorizer(stop_words="english", min_df=5)
        dtm = vec.fit_transform(df[text_col].dropna().astype(str))

        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)

        words = vec.get_feature_names_out()
        for i, topic in enumerate(lda.components_):
            top_words = [words[j] for j in topic.argsort()[:-6:-1]]
            st.markdown(f"**Topic {i+1}:** {', '.join(top_words)}")
    except:
        st.warning("Not enough data for topic modeling.")

# =================================================
# STUDENT VIEW
# =================================================
with student_tab:
    st.markdown("### 📌 Key Insights")

    if neg_pct > 30:
        st.markdown("<div class='section bad'>⚠️ Students are largely dissatisfied in this area.</div>", unsafe_allow_html=True)
    elif neg_pct > 15:
        st.markdown("<div class='section warn'>⚠️ Mixed feedback — improvements recommended.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section good'>✅ Students are generally satisfied.</div>", unsafe_allow_html=True)

    with st.expander("ℹ️ How to interpret this?"):
        st.write(
            "Insights are based on sentiment polarity derived from student feedback using NLP."
        )

# =============================
# PROFESSIONAL ENDING
# =============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Advanced NLP Dashboard | Built with Streamlit & Python</p>",
    unsafe_allow_html=True
)
