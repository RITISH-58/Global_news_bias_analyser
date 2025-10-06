# app.py
import os
import re
import json
import math
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

import streamlit as st

# Article extraction
from newspaper import Article
from bs4 import BeautifulSoup

# Sentiment & simple bias heuristic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For small tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# --- Ensure NLTK data (first run will download) ---
nltk.download('punkt')

# --- Load keys ---
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() in ("1", "true", "yes")

# --- Globals & helpers ---
analyzer = SentimentIntensityAnalyzer()

# A small list of opinion words that often indicate subjective language.
OPINION_WORDS = {
    "outrageous","disastrous","horrific","unacceptable","unjust","heroic","brilliant",
    "terrible","amazing","ridiculous","shocking","catastrophic","terrible","appalling",
    "sensational","stunning","baffling","outrage","scandalous","biased","extremist",
    "alarming","dangerous","grossly"
}

LEFT_KEYWORDS = {"liberal","progressive","democrat","left-wing","leftist","socialist","pro-choice"}
RIGHT_KEYWORDS = {"conservative","republican","right-wing","rightist","traditionalist","pro-life"}

# basic clean
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# Agent 1: normalize
def agent1_normalize_query(query: str) -> str:
    q = query.strip()
    q = re.sub(r"[^\w\s\-\']", " ", q)  # keep words, spaces, hyphens, apostrophes
    q = re.sub(r"\s+", " ", q)
    q = q.lower()
    return q

# Agent 2: fetch articles using NewsAPI and extract full text
def agent2_fetch_articles(query: str, max_articles: int = 10, from_days: int = 7) -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not found in environment. Put it in .env")
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.utcnow() - timedelta(days=from_days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": min(max_articles, 100),
        "apiKey": NEWSAPI_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hits = data.get("articles", [])[:max_articles]

    results = []
    for a in hits:
        art = {
            "title": a.get("title"),
            "url": a.get("url"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
            "description": a.get("description"),
            "urlToImage": a.get("urlToImage"),
            "raw_content": None
        }
        # Try to download full article via newspaper
        try:
            article = Article(a.get("url"))
            article.download()
            article.parse()
            text = article.text
            text = clean_text(text)
            if text and len(text.split()) > 30:
                art["raw_content"] = text
            else:
                art["raw_content"] = a.get("content") or a.get("description") or ""
        except Exception:
            # fallback scrape
            try:
                rr = requests.get(a.get("url"), timeout=15)
                soup = BeautifulSoup(rr.text, "html.parser")
                paragraphs = [p.get_text() for p in soup.find_all("p")]
                text = " ".join(paragraphs)
                text = clean_text(text)
                art["raw_content"] = text or a.get("content") or a.get("description") or ""
            except Exception:
                art["raw_content"] = a.get("content") or a.get("description") or ""
        results.append(art)
    return results

# Agent 3: analyze bias (local heuristic) - returns score 0..100, reasons, flagged sentences
def agent3_analyze_bias(article_text: str) -> Dict[str, Any]:
    text = clean_text(article_text)
    if not text:
        return {"score": 0, "reasons": ["No text available"], "leaning": "unknown", "top_flagged": []}

    # Sentiment
    s = analyzer.polarity_scores(text)
    compound = s["compound"]  # -1 .. 1
    compound_abs = abs(compound)

    # Opinion-word density
    tokens = [w.lower() for w in word_tokenize(text)]
    total_words = max(1, len(tokens))
    opinion_count = sum(1 for t in tokens if t in OPINION_WORDS)
    opinion_density = opinion_count / total_words  # small fractional

    # Heuristic combined score
    # compound_abs weighted higher; opinion_density scaled up a bit
    opinion_factor = min(1.0, opinion_density * 5.0)
    combined = compound_abs * 0.7 + opinion_factor * 0.3
    bias_score = int(min(100, max(0, combined * 100)))

    # political leaning guess via keyword counts
    left_ct = sum(text.lower().count(w) for w in LEFT_KEYWORDS)
    right_ct = sum(text.lower().count(w) for w in RIGHT_KEYWORDS)
    if left_ct == right_ct == 0:
        leaning = "unknown"
    elif left_ct > right_ct:
        leaning = "left-leaning"
    elif right_ct > left_ct:
        leaning = "right-leaning"
    else:
        leaning = "mixed"

    # Find flagged sentences (high sentiment or containing opinion words)
    sentences = sent_tokenize(text)
    flagged = []
    for sent in sentences:
        ss = analyzer.polarity_scores(sent)
        if abs(ss["compound"]) > 0.4 or any(w.lower() in OPINION_WORDS for w in word_tokenize(sent)):
            flagged.append({"sentence": sent, "sentiment": ss})

    # Build reasons
    reasons = []
    reasons.append(f"Neutrality risk from emotional tone (compound sentiment magnitude {compound_abs:.2f}).")
    if opinion_count > 0:
        reasons.append(f"{opinion_count} opinion words detected (opinion density {opinion_density:.3f}).")
    if leaning != "unknown":
        reasons.append(f"Keywords indicate possible {leaning}.")
    if not reasons:
        reasons.append("No obvious bias indicators found.")

    return {
        "score": bias_score,
        "reasons": reasons,
        "leaning": leaning,
        "top_flagged": flagged[:6]
    }

# Optional: placeholder Gemini call (user must replace endpoint/payload per their Gemini setup)
def call_gemini_analyze(prompt_text: str) -> Dict[str, Any]:
    """
    Placeholder example. If you have Gemini/Vertex AI keys and client, replace this with an actual call.
    This function shows a general structure: send prompt_text and parse JSON response.
    """
    if not GEMINI_ENABLED or not GEMINI_API_KEY:
        raise RuntimeError("Gemini not enabled/configured. Set GEMINI_ENABLED=true and GEMINI_API_KEY in .env")
    # This is a placeholder - the exact REST endpoint and payload depend on your setup (Google Vertex AI / Gemini).
    GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": {
            "text": prompt_text
        },
        "maxOutputTokens": 600
    }
    r = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # You must parse the response format that your API returns.
    return data

# Agent 4: create "zero-bias" article by combining factual sentences from multiple sources
def agent4_generate_neutral_article(articles: List[Dict[str, Any]], bias_results: List[Dict[str, Any]], topic: str, use_gemini: bool=False) -> Dict[str, Any]:
    """
    If use_gemini==True and GEMINI is configured, you can send a prompt to Gemini to rewrite.
    Otherwise we create a neutral article using a simple aggregator: pick low-sentiment sentences from all articles,
    order them, and create a short factual summary + references.
    """
    # If Gemini is enabled and requested, build a prompt and call it (placeholder)
    if use_gemini and GEMINI_ENABLED and GEMINI_API_KEY:
        # Build prompt: include top article texts and ask for neutral rewrite with references
        combined_text = "\n\n".join([a.get("raw_content","")[:3000] for a in articles])
        prompt = (
            "You are a neutral journalist. Given the following combined article texts about the topic:\n\n"
            f"TOPIC: {topic}\n\n"
            "ARTICLES:\n" + combined_text + "\n\n"
            "TASK: Write a single neutral, balanced article on the topic. "
            "Include brief mention of differing viewpoints if they exist, a short conclusion, and list the references (title + URL). "
            "Return the article followed by a JSON block with keys: title, references (list of {title,url}), images (list of image urls).\n"
        )
        gemini_response = call_gemini_analyze(prompt)  # user must ensure this function matches their API
        # parse gemini_response depending on returned format - here we are simply returning raw response
        return {"mode": "gemini", "raw": gemini_response}

    # Local neutralizer path (no Gemini)
    sentences = []
    references = []
    images = []
    for a in articles:
        text = a.get("raw_content") or ""
        if not text: 
            continue
        for s in sent_tokenize(text):
            s_clean = s.strip()
            if len(s_clean.split()) < 6:
                continue
            # measure sentiment magnitude of sentence
            ss = analyzer.polarity_scores(s_clean)
            mag = abs(ss["compound"])
            sentences.append({"sent": s_clean, "sent_mag": mag, "source": a.get("source"), "url": a.get("url")})
        if a.get("url"):
            references.append({"title": a.get("title") or a.get("source") or "source", "url": a.get("url")})
        if a.get("urlToImage"):
            images.append(a.get("urlToImage"))

    # pick factual sentences = those with low sentiment magnitude
    # sort ascending by sent_mag to get most factual first
    sentences_sorted = sorted(sentences, key=lambda x: x["sent_mag"])
    # choose top N sentences but avoid duplicates and too many from the same source
    chosen = []
    seen_text = set()
    source_count = {}
    for s in sentences_sorted:
        if len(chosen) >= 30:
            break
        t = s["sent"].strip()
        if t in seen_text:
            continue
        src = s.get("source") or "unknown"
        if source_count.get(src, 0) >= 6:  # avoid >6 sentences from same source
            continue
        chosen.append(s)
        seen_text.add(t)
        source_count[src] = source_count.get(src, 0) + 1

    # build neutral article: title + lead + paragraphs (group sentences into small paragraphs)
    title = f"Neutral overview: {topic.capitalize()}"
    lead = ""
    if chosen:
        lead = chosen[0]["sent"]
    else:
        lead = f"This is a neutral overview of {topic} based on multiple reporting sources."

    paragraphs = []
    cur_para = []
    for i, s in enumerate(chosen[1:], start=1):
        cur_para.append(s["sent"])
        if i % 4 == 0:
            paragraphs.append(" ".join(cur_para))
            cur_para = []
    if cur_para:
        paragraphs.append(" ".join(cur_para))

    article_text = f"{lead}\n\n" + "\n\n".join(paragraphs)
    # prepare references: deduplicate by URL
    seen_urls = set()
    refs_clean = []
    for r in references:
        if r["url"] not in seen_urls:
            refs_clean.append(r)
            seen_urls.add(r["url"])

    return {
        "mode": "local",
        "title": title,
        "article": article_text,
        "references": refs_clean,
        "images": images[:6]
    }

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Global News Bias Analyser", layout="wide")
st.title("Global News Bias Analyser using AI-Agents (Streamlit)")

st.markdown("""
Enter a topic or query, then press **Analyze**.
This app:
- Agent 1: cleans & normalizes your query
- Agent 2: collects articles (NewsAPI) and extracts full text
- Agent 3: analyzes bias (local heuristic; optionally Gemini)
- Agent 4: produces a neutral article (aggregator; optionally Gemini rewrite)
""")

with st.sidebar:
    st.header("Settings")
    max_articles = st.number_input("Max articles to fetch", min_value=3, max_value=50, value=8, step=1)
    days_back = st.number_input("Search last N days", min_value=1, max_value=30, value=7)
    use_gemini_ui = st.checkbox("Use Gemini for analysis & generation (if configured)", value=GEMINI_ENABLED)
    st.markdown("**Environment keys** (from .env):")
    st.write(f"NEWSAPI: {'configured' if NEWSAPI_KEY else 'NOT configured'}")
    st.write(f"GEMINI: {'configured' if GEMINI_API_KEY and GEMINI_ENABLED else 'NOT configured / disabled'}")
    st.caption("Get a NewsAPI key at https://newsapi.org/")

query = st.text_input("What news topic would you like to analyze?", value="climate change policy")

if st.button("Analyze"):
    qnorm = agent1_normalize_query(query)
    st.info(f"Normalized query: **{qnorm}**")
    try:
        with st.spinner("Fetching articles (Agent 2)..."):
            articles = agent2_fetch_articles(qnorm, max_articles=int(max_articles), from_days=int(days_back))
        if not articles:
            st.warning("No articles found for this query. Try a different query or increase max articles / days.")
        else:
            st.success(f"Fetched {len(articles)} articles.")
            # Show fetched articles list
            for i, a in enumerate(articles, start=1):
                st.markdown(f"**{i}. {a.get('title')}**  \nSource: {a.get('source')}  •  {a.get('publishedAt')}")
                if a.get("urlToImage"):
                    st.image(a.get("urlToImage"), width=300)
                st.write(a.get("description") or "")
                st.write(f"URL: {a.get('url')}")
                st.write("---")

            # Agent 3: analyze every article
            st.subheader("Bias Analysis (Agent 3)")
            bias_results = []
            with st.spinner("Analyzing bias for each article..."):
                for a in articles:
                    text = a.get("raw_content") or a.get("description") or ""
                    if use_gemini_ui and GEMINI_ENABLED:
                        # Construct a short prompt for Gemini (placeholder behavior)
                        try:
                            prompt = f"Analyze bias in the following text. Return JSON with fields: score (0-100), reasons (list), leaning (left/right/center/unknown). Text:\n\n{text[:3000]}"
                            gem_res = call_gemini_analyze(prompt)
                            # NOTE: parsing depends on your Gemini response format. For now, fallback to local heuristic if parsing fails.
                            parsed = agent3_analyze_bias(text)
                            bias_results.append(parsed)
                        except Exception as e:
                            # fallback to local
                            parsed = agent3_analyze_bias(text)
                            bias_results.append(parsed)
                    else:
                        parsed = agent3_analyze_bias(text)
                        bias_results.append(parsed)

            # Show table of results
            import pandas as pd
            rows = []
            for a, b in zip(articles, bias_results):
                rows.append({
                    "title": a.get("title"),
                    "source": a.get("source"),
                    "bias_score": b["score"],
                    "leaning": b["leaning"],
                    "reasons": "; ".join(b["reasons"])
                })
            df = pd.DataFrame(rows)
            st.dataframe(df)

            avg_bias = int(sum(b["score"] for b in bias_results) / max(1, len(bias_results)))
            st.metric("Aggregated average bias score (0=least biased)", value=f"{avg_bias} / 100")

            # Agent 4: generate neutral article
            st.subheader("Generated neutral article (Agent 4)")
            with st.spinner("Generating neutral article..."):
                neutral = agent4_generate_neutral_article(articles, bias_results, topic=qnorm, use_gemini=use_gemini_ui)
            if neutral.get("mode") == "gemini":
                st.write("Gemini returned raw response (see data). Please parse per your Gemini output format.")
                st.write(neutral.get("raw"))
            else:
                st.markdown(f"### {neutral.get('title')}")
                st.write(neutral.get("article"))
                st.markdown("**References**")
                for r in neutral.get("references", [])[:10]:
                    st.write(f"- [{r.get('title')}]({r.get('url')})")
                if neutral.get("images"):
                    st.markdown("**Images**")
                    cols = st.columns(3)
                    for i, img in enumerate(neutral.get("images")):
                        try:
                            cols[i % 3].image(img, width=200)
                        except Exception:
                            cols[i % 3].write(img)

            # Downloads: CSV of results and neutral article
            st.subheader("Export")
            st.download_button("Download analysis (JSON)", data=json.dumps({
                "query": qnorm,
                "articles": articles,
                "bias_results": bias_results,
                "neutral_article": neutral
            }, indent=2), file_name="news_bias_analysis.json", mime="application/json")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)