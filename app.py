import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import joblib
import os
from textblob import TextBlob
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Persona Engine", page_icon="🧠", layout="wide")

st.title("Persona Engine — Full Demo")
st.caption("Part 1: Drift Detector  |  Part 2: Intent Classifier  |  Part 3: RAG Conflict Resolver")

# ── LOAD DATA (cached so it runs only once) ───────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/conversations.csv", header=None, names=["conv"])
    df["day"] = (df.index // 1000) + 1
    return df

@st.cache_resource
def build_rag(df):
    chunks = []
    for _, row in df.iterrows():
        text = str(row["conv"])
        day  = int(row["day"])
        pos  = len(re.findall(r'\b(love|happy|great|excited|good|wonderful|amazing)\b', text, re.I))
        neg  = len(re.findall(r'\b(hate|angry|sad|frustrated|upset|fight|terrible|awful)\b', text, re.I))
        chunks.append({"text": text, "day": day, "emotional_weight": pos + neg * 1.5})
    texts      = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf_mat  = vectorizer.fit_transform(texts)
    return chunks, vectorizer, tfidf_mat

@st.cache_resource
def load_or_train_classifier(df):
    if os.path.exists("intent_classifier.pkl"):
        return joblib.load("intent_classifier.pkl")
    training_data = [
        ("remind me to call mom tomorrow", "reminder"),
        ("don't let me forget the meeting", "reminder"),
        ("set a reminder for my appointment", "reminder"),
        ("please remind me at 6pm", "reminder"),
        ("notify me before the meeting", "reminder"),
        ("I need to remember to pay the bill", "reminder"),
        ("I feel so alone lately", "emotional-support"),
        ("I am really stressed about work", "emotional-support"),
        ("I am going through a tough time", "emotional-support"),
        ("I feel anxious and overwhelmed", "emotional-support"),
        ("I had a really bad day", "emotional-support"),
        ("I am scared about the future", "emotional-support"),
        ("can you write an email to my boss", "action-item"),
        ("search for the best restaurants", "action-item"),
        ("make a list of things to buy", "action-item"),
        ("book a flight to Mumbai", "action-item"),
        ("find me a recipe for pasta", "action-item"),
        ("summarize this document", "action-item"),
        ("hey how are you doing", "small-talk"),
        ("tell me a joke", "small-talk"),
        ("nice weather today", "small-talk"),
        ("good morning hope you have a great day", "small-talk"),
        ("lets just chat for a bit", "small-talk"),
        ("what is your name", "small-talk"),
        ("xkcd flux capacitor override", "unknown"),
        ("zzzz blah blah nonsense", "unknown"),
        ("nothing just testing", "unknown"),
    ]
    for _, row in df.sample(min(200, len(df)), random_state=42).iterrows():
        lines = [l.replace("User 1:", "").strip()
                 for l in str(row["conv"]).split("\n") if l.startswith("User 1:")]
        for line in lines[:2]:
            if line and len(line) > 5:
                training_data.append((line, "small-talk"))
    texts, labels = zip(*training_data)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LinearSVC(max_iter=2000)),
    ])
    model.fit(texts, labels)
    joblib.dump(model, "intent_classifier.pkl")
    return model

df                       = load_data()
chunks, vectorizer, tmat = build_rag(df)
classifier               = load_or_train_classifier(df)
max_day                  = int(df["day"].max())

# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Part 1 — Persona Drift", "Part 2 — Intent Classifier", "Part 3 — RAG Resolver"])

# ══ PART 1 ════════════════════════════════════════════════════
with tab1:
    st.header("Persona Drift Detector")
    st.write("Tracks how the user's tone changes across days and detects what triggered each drift.")

    def detect_tone(text):
        blob     = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        tl       = text.lower()
        frustrated_hits = sum(1 for k in ["why","ugh","again","still","terrible","awful","hate","annoying","can't","won't"] if k in tl)
        playful_hits    = sum(1 for k in ["haha","lol","hehe","hilarious","joke","funny","laugh","silly"] if k in tl)
        formal_hits     = sum(1 for k in ["therefore","however","regarding","sincerely","furthermore","accordingly"] if k in tl)
        curious_hits    = sum(1 for k in ["how","what","why","when","could you","tell me","explain","wondering","?"] if k in tl)
        if frustrated_hits >= 2 and polarity < 0: return "frustrated"
        if playful_hits >= 2:                      return "playful"
        if formal_hits >= 1:                       return "formal"
        if curious_hits >= 2:                      return "curious"
        if polarity > 0.3:                         return "positive"
        if polarity < -0.2:                        return "frustrated"
        return "casual"

    def detect_trigger(text):
        people = re.findall(r'\bmy\s+(\w+)\b', text, re.I)
        topics = ["work","sister","family","boss","project","deadline","school","money","health","friend","meeting"]
        if people:           return f"person/relation: '{people[0]}'"
        found = [t for t in topics if t in text.lower()]
        if found:            return f"topic: '{found[0]}'"
        return "unknown trigger"

    @st.cache_data
    def compute_timeline(_df):
        _df = _df.copy()
        def ex(t):
            lines = str(t).split("\n")
            return " ".join(l.replace("User 1:","").strip() for l in lines if l.startswith("User 1:"))
        _df["user_text"] = _df["conv"].apply(ex)
        _df["tone"]      = _df["user_text"].apply(detect_tone)
        day_tones        = _df.groupby("day")["tone"].agg(lambda x: x.mode()[0]).reset_index()
        timeline = []
        prev = None
        for _, row in day_tones.iterrows():
            entry = {"day": int(row["day"]), "tone": row["tone"], "drift": False, "trigger": None}
            if prev and row["tone"] != prev:
                entry["drift"]   = True
                day_text         = " ".join(_df[_df["day"] == row["day"]]["user_text"])
                entry["trigger"] = detect_trigger(day_text)
            timeline.append(entry)
            prev = row["tone"]
        return timeline, _df["tone"].value_counts()

    timeline, tone_dist = compute_timeline(df)

    tone_colors = {
        "curious":   "#378ADD",
        "casual":    "#1D9E75",
        "playful":   "#D4537E",
        "formal":    "#534AB7",
        "frustrated":"#E24B4A",
        "positive":  "#639922",
    }

    st.subheader("Day-by-day tone timeline")
    for t in timeline:
        color = tone_colors.get(t["tone"], "#888")
        drift_badge = f'<span style="background:#FAEEDA;color:#633806;padding:2px 8px;border-radius:6px;font-size:12px;margin-left:10px">DRIFT — {t["trigger"]}</span>' if t["drift"] else ""
        st.markdown(
            f'<div style="padding:8px 14px;margin-bottom:6px;border-radius:8px;border:0.5px solid #ddd;display:flex;align-items:center;gap:12px">'
            f'<span style="font-weight:500;min-width:50px">Day {t["day"]}</span>'
            f'<span style="background:{color};color:white;padding:3px 12px;border-radius:12px;font-size:13px">{t["tone"]}</span>'
            f'{drift_badge}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Tone distribution across all messages")
    st.bar_chart(tone_dist)

# ══ PART 2 ════════════════════════════════════════════════════
with tab2:
    st.header("Offline Intent Classifier")
    st.write("Type any message — classified instantly on CPU with no API calls.")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_msg = st.text_input("Type a message:", placeholder="e.g. remind me to call mom tomorrow")
    with col2:
        st.write("")
        st.write("")
        classify_btn = st.button("Classify", use_container_width=True)

    if classify_btn and user_msg:
        start      = time.time()
        pred       = classifier.predict([user_msg])[0]
        scores_raw = classifier.decision_function([user_msg])[0]
        confidence = float(max(scores_raw))
        latency    = (time.time() - start) * 1000
        if confidence < 0.2:
            pred = "unknown"

        intent_colors = {
            "reminder":         "#185FA5",
            "emotional-support":"#993556",
            "action-item":      "#3B6D11",
            "small-talk":       "#534AB7",
            "unknown":          "#5F5E5A",
        }
        color = intent_colors.get(pred, "#888")

        c1, c2, c3 = st.columns(3)
        c1.metric("Intent",     pred)
        c2.metric("Confidence", f"{confidence:.2f}")
        c3.metric("Latency",    f"{latency:.1f} ms")

        st.markdown(
            f'<div style="background:{color};color:white;padding:14px 20px;border-radius:10px;font-size:18px;font-weight:500;margin-top:10px">'
            f'{pred.upper()}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Batch test")
    examples = [
        "remind me to buy groceries tomorrow",
        "I feel really sad and lonely today",
        "can you search for flights to Delhi",
        "hey what is up how are you",
        "set an alarm for 7am",
        "I am so stressed about my exams",
        "make a list of things I need to pack",
    ]
    rows = []
    for msg in examples:
        pred       = classifier.predict([msg])[0]
        scores_raw = classifier.decision_function([msg])[0]
        conf       = float(max(scores_raw))
        rows.append({"Message": msg, "Intent": pred, "Confidence": round(conf, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══ PART 3 ════════════════════════════════════════════════════
with tab3:
    st.header("RAG Conflict Resolver")
    st.write("Ask what was mentioned — retrieves relevant chunks, ranks by recency + emotion, flags contradictions.")

    query = st.text_input("Ask a question:", placeholder="Did I mention anything about my sister?")
    search_btn = st.button("Search", use_container_width=False)

    if search_btn and query:
        keyword = query.split()[-1].rstrip("?").lower()

        q_vec   = vectorizer.transform([query])
        sims    = cosine_similarity(q_vec, tmat)[0]
        top_idx = np.argsort(sims)[::-1][:15]
        hits    = [dict(chunks[i], similarity=float(sims[i])) for i in top_idx if sims[i] > 0.005]

        ranked = sorted(hits, key=lambda c: (
            0.40 * c["day"] / max_day +
            0.35 * min(c["emotional_weight"] / 10.0, 1.0) +
            0.25 * c["similarity"]
        ), reverse=True)[:5]

        # Contradiction detection
        polarities, flags = [], []
        for c in ranked:
            sentences = [s.strip() for s in c["text"].split(".") if keyword in s.lower() and len(s.strip()) > 5]
            if sentences:
                pol = TextBlob(" ".join(sentences)).sentiment.polarity
                polarities.append({"day": c["day"], "polarity": round(pol, 3), "snippet": sentences[0][:120]})
        for i in range(1, len(polarities)):
            p, q2 = polarities[i-1], polarities[i]
            if p["polarity"] * q2["polarity"] < 0:
                flags.append(f"Day {p['day']} was {'positive' if p['polarity']>0 else 'negative'} ({p['polarity']}) but Day {q2['day']} was {'positive' if q2['polarity']>0 else 'negative'} ({q2['polarity']})")

        # Answer
        parts = []
        for c in ranked[:3]:
            sentences = [s.strip() for s in c["text"].split(".") if keyword in s.lower() and len(s.strip()) > 10]
            if sentences:
                parts.append(f"(Day {c['day']}) {sentences[0][:150]}")
        answer = " | ".join(parts) if parts else f"No specific mentions of '{keyword}' found."

        st.subheader("Answer")
        st.info(answer)

        if flags:
            st.subheader("Contradictions detected")
            for f in flags:
                st.warning(f"!! {f}")
        else:
            st.success("No contradictions found across chunks.")

        st.subheader("Top ranked chunks")
        rows = []
        for c in ranked:
            score = round(0.40 * c["day"]/max_day + 0.35 * min(c["emotional_weight"]/10,1) + 0.25 * c["similarity"], 4)
            rows.append({
                "Day":            c["day"],
                "Final Score":    score,
                "Similarity":     round(c["similarity"], 4),
                "Emotional Wt":   round(c["emotional_weight"], 1),
                "Recency":        round(c["day"] / max_day, 3),
                "Snippet":        c["text"][:80] + "...",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Sentiment per chunk")
        if polarities:
            st.dataframe(pd.DataFrame(polarities), use_container_width=True)
        else:
            st.write("No sentences with that keyword found in top chunks.")