# feedback_classifier_app.py

import os
import streamlit as st
import pandas as pd
import time
import schedule
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
import random
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import re

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# Gemini configuration
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("models/gemini-2.0-flash")

def classify_feedback(feedback_text):
    prompt = f"""
    Classify the following user feedback into one of the categories: Feature Request, Bug Report, Compliment, Complaint, or Other.
    Feedback: "{feedback_text}"
    Respond with only the category name.
    """
    try:
        response = model.generate_content(prompt)
        time.sleep(random.uniform(2.5, 4.5))  # Throttle to avoid rate limits
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def analyze_sentiment(feedback_text):
    blob = TextBlob(feedback_text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(text, top_n=10):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", header=None)[0])
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = RECIPIENT_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        print("Failed to send email:", e)

# Feedback classification function with scheduling
@st.cache_data
def run_classification(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "Feedback" not in df.columns:
        st.error("CSV must contain a 'Feedback' column.")
        return None
    df["Category"] = df["Feedback"].apply(classify_feedback)
    df["Sentiment"] = df["Feedback"].apply(analyze_sentiment)
    summary = df["Category"].value_counts().to_string()
    send_email("Daily Feedback Classification Summary", summary)
    return df

# Schedule function
if "scheduler_started" not in st.session_state:
    def job():
        if "uploaded_file" in st.session_state:
            df = run_classification(st.session_state.uploaded_file)
            st.session_state["classified_df"] = df
    schedule.every().day.at("12:00").do(job)
    st.session_state.scheduler_started = True

# UI
st.title("Product Feedback Classifier Agent")

uploaded_file = st.file_uploader("Upload CSV File with 'Feedback' column", type="csv")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    with st.spinner("Classifying feedback using Gemini..."):
        df_result = run_classification(uploaded_file)
        if df_result is not None:
            st.session_state.classified_df = df_result
            st.success("Classification complete!")
            st.dataframe(df_result)

            st.markdown("### Category Distribution 📊")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df_result, x="Category", order=df_result["Category"].value_counts().index, ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            st.markdown("### Sentiment Analysis 📈")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df_result, x="Sentiment", order=["Positive", "Neutral", "Negative"], ax=ax2)
            st.pyplot(fig2)

            st.markdown("### Advanced Analytics 🔍")
            fig3 = px.sunburst(df_result, path=["Category", "Sentiment"], title="Category vs Sentiment")
            st.plotly_chart(fig3)

            fig4 = px.histogram(df_result, x="Category", color="Sentiment", barmode="group", title="Category by Sentiment")
            st.plotly_chart(fig4)

            st.markdown("### Keyword Word Cloud from Feedback ☁️")
            all_feedback_text = " ".join(df_result["Feedback"].dropna().astype(str))
            
            # Load stopwords
            stopwords_set = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", header=None)[0])
            words = re.findall(r'\b\w+\b', all_feedback_text.lower())
            filtered_text = " ".join([word for word in words if word not in stopwords_set and len(word) > 2])
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(filtered_text)
            
            fig6, ax6 = plt.subplots(figsize=(10, 5))
            ax6.imshow(wordcloud, interpolation="bilinear")
            ax6.axis("off")
            st.pyplot(fig6)

# Run scheduler loop manually (simulated cron)
for _ in range(3):
    schedule.run_pending()
    time.sleep(1)

if "classified_df" in st.session_state:
    st.download_button("Download Results as CSV", data=st.session_state.classified_df.to_csv(index=False), file_name="classified_feedback.csv")
