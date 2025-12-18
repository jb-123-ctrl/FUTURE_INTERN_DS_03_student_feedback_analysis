import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    layout="wide"
)

# ==================================================
# THEME (Sea Blue + Green)
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E0F7FA, #E8F5E9);
    color: #1F2933;
}
h1, h2, h3 {
    color: #1F2933;
}
.insight-box {
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.good { background-color: #D1FAE5; }
.warn { background-color: #FEF3C7; }
.bad { background-color: #FEE2E2; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.markdown("<h1 style='text-align:center;'>🎓 Student Feedback Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Advanced NLP Dashboard with Insights & Recommendations</p>", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    return pd.read_excel("dataset/finalDataset0.2.xlsx")

df = load_data()

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
st.sidebar.header("🔍 Controls")

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
    list(category_map.keys()),
    help="Choose which feedback area to analyze"
)

text_column = category_map[selected_category]

# ==================================================
# SENTIMENT FUNCTIONS
# ==================================================
def get_polarity(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

def get_sentiment_label(score):
    if score > 0.5:
        return "Strong Positive"
    elif score > 0:
        return "Mild Positive"
    elif score == 0:
        return "Neutral"
    elif score > -0.5:
        return "Mild Negative"
    else:
        return "Strong Negative"

df["Polarity"] = df[text_column].apply(get_polarity)
df["Sentiment_Level"] = df["Polarity"].apply(get_sentiment_label)

# ==================================================
# KPI CARDS
# ==================================================
st.markdown("## 📊 Key Performance Indicators")

total = len(df)
pos_pct = round((df["Polarity"] > 0).sum() / total * 100, 1)
neg_pct = round((df["Polarity"] < 0).sum() / total * 100, 1)
risk_count = (df["Polarity"] < -0.3).sum()

c1, c2, c3 = st.columns(3)
c1.metric("😊 Satisfaction Score", f"{pos_pct}%", help="Percentage of positive feedback")
c2.metric("⚠️ Risk Area Count", risk_count, help="Strongly negative responses")
c3.metric("📉 Negative Trend %", f"{neg_pct}%", help="Percentage of negative feedback")

# ==================================================
# AUTO RECOMMENDATIONS
# ==================================================
st.markdown("## 🤖 Automated Recommendations")

if neg_pct > 30:
    st.markdown("<div class='insight-box bad'>⚠️ Immediate improvement needed in this category.</div>", unsafe_allow_html=True)
elif neg_pct > 15:
    st.markdown("<div class='insight-box warn'>⚠️ Moderate dissatisfaction detected. Review feedback.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='insight-box good'>✅ This area is performing well.</div>", unsafe_allow_html=True)

with st.expander("ℹ️ How to interpret this?"):
    st.write("Recommendations are generated based on the percentage and severity of negative sentiment.")

# ==================================================
# SENTIMENT DISTRIBUTION (PIE)
# ==================================================
st.markdown("## 🥧 Sentiment Distribution")

sent_counts = df["Sentiment_Level"].value_counts()

pie_fig = px.pie(
    values=sent_counts.values,
    names=sent_counts.index,
    hole=0.35,
    color_discrete_sequence=px.colors.qualitative.Set2
)

pie_fig.update_traces(pull=[0.05]*len(sent_counts))
st.plotly_chart(pie_fig, use_container_width=True)

# ==================================================
# HEATMAP (ADVANCED)
# ==================================================
st.markdown("## 🔥 Sentiment Heatmap (Aspect × Severity)")

heat_df = df["Sentiment_Level"].value_counts().reset_index()
heat_df.columns = ["Sentiment", "Count"]

fig, ax = plt.subplots()
sns.heatmap(
    heat_df[["Count"]].T,
    annot=True,
    cmap="YlGnBu",
    fmt="d",
    ax=ax
)

st.pyplot(fig)

with st.expander("ℹ️ How to interpret this heatmap?"):
    st.write("Darker colors indicate higher concentration of feedback in that sentiment category.")

# ==================================================
# WORD CLOUD
# ==================================================
st.markdown("## ☁️ Common Feedback Themes")

text_data = " ".join(df[text_column].dropna().astype(str))

wordcloud = WordCloud(
    width=900,
    height=400,
    background_color="white",
    colormap="summer"
).generate(text_data)

fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
ax_wc.imshow(wordcloud)
ax_wc.axis("off")
st.pyplot(fig_wc)

# ==================================================
# TOPIC MODELING (LDA)
# ==================================================
st.markdown("## 🧠 Topic Modeling (LDA)")

try:
    vectorizer = CountVectorizer(stop_words="english", min_df=5)
    dtm = vectorizer.fit_transform(df[text_column].dropna().astype(str))

    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        words = [feature_names[j] for j in topic.argsort()[:-7:-1]]
        st.markdown(f"**Topic {i+1}:** {', '.join(words)}")

except:
    st.warning("Topic modeling could not be generated due to limited data.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown("<p style='text-align:center;font-size:12px;'>Advanced NLP Dashboard | Streamlit</p>", unsafe_allow_html=True)
