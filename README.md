# 🌍 Global News Bias Analyzer

An AI-powered Streamlit web app that analyzes global news articles for potential bias using automated agents.  
It collects recent news about any topic, evaluates sentiment and bias-indicating language, and generates a neutral summary.

---

## 🧠 Overview

This app uses *four AI-inspired agents*:

1. *Agent 1 – Query Normalizer:*  
   Cleans and standardizes the user’s search topic.

2. *Agent 2 – News Fetcher:*  
   Collects recent news articles using the [NewsAPI](https://newsapi.org/).

3. *Agent 3 – Bias Analyzer:*  
   Uses sentiment analysis and heuristic rules to estimate bias.

4. *Agent 4 – Neutral Summarizer:*  
   Generates a neutral, balanced summary (optionally via Google Gemini API).

---

## ⚙ Tech Stack

- *Frontend & Backend:* Streamlit  
- *APIs:* NewsAPI, Gemini API (optional)  
- *NLP Tools:* NLTK, VaderSentiment, Newspaper3k, BeautifulSoup4  
- *Language:* Python 3.8+

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/global-news-bias-analyzer.git
cd global-news-bias-analyzer
