import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# ── 1. LOAD AND BUILD CHUNKS ──────────────────────────────────
print("Loading conversations ...")
df = pd.read_csv("data/conversations.csv", header=None, names=["conv"])
df["day"] = (df.index // 1000) + 1

def make_chunks(df):
    chunks = []
    for _, row in df.iterrows():
        text = str(row["conv"])
        day  = int(row["day"])

        positive = len(re.findall(
            r'\b(love|happy|great|excited|good|wonderful|amazing)\b', text, re.I))
        negative = len(re.findall(
            r'\b(hate|angry|sad|frustrated|upset|fight|terrible|awful)\b', text, re.I))
        emotional_weight = positive + negative * 1.5

        chunks.append({
            "text":             text,
            "day":              day,
            "emotional_weight": emotional_weight,
        })
    return chunks

chunks = make_chunks(df)
print(f"Total chunks built: {len(chunks)}")

# ── 2. BUILD TF-IDF INDEX ─────────────────────────────────────
print("Building TF-IDF index ...")
texts      = [c["text"] for c in chunks]
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
tfidf_mat  = vectorizer.fit_transform(texts)
max_day    = df["day"].max()

# ── 3. RETRIEVE TOP CHUNKS ────────────────────────────────────
def retrieve(query: str, top_k: int = 10):
    q_vec = vectorizer.transform([query])
    sims  = cosine_similarity(q_vec, tfidf_mat)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        if sims[i] > 0.005:
            results.append({**chunks[i], "similarity": float(sims[i])})
    return results

# ── 4. RANK BY RECENCY + EMOTIONAL WEIGHT + SIMILARITY ────────
def rank_chunks(hits):
    ranked = []
    for c in hits:
        recency_score   = c["day"] / max_day
        emotional_score = min(c["emotional_weight"] / 10.0, 1.0)
        sim_score       = c["similarity"]

        final_score = (0.40 * recency_score +
                       0.35 * emotional_score +
                       0.25 * sim_score)

        ranked.append({**c, "final_score": round(final_score, 4)})

    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)

# ── 5. DETECT CONTRADICTIONS ──────────────────────────────────
def detect_contradictions(top_chunks, keyword):
    polarities = []
    for c in top_chunks:
        sentences = [s.strip() for s in c["text"].split(".")
                     if keyword.lower() in s.lower() and len(s.strip()) > 5]
        if sentences:
            pol = TextBlob(" ".join(sentences)).sentiment.polarity
            polarities.append({
                "day":      c["day"],
                "polarity": round(pol, 3),
                "snippet":  sentences[0][:120],
            })

    flags = []
    for i in range(1, len(polarities)):
        prev = polarities[i - 1]
        curr = polarities[i]
        if prev["polarity"] * curr["polarity"] < 0:
            flags.append({
                "between_days": [prev["day"], curr["day"]],
                "type":         "sentiment contradiction",
                "detail": (
                    f"Day {prev['day']} was "
                    f"{'positive' if prev['polarity'] > 0 else 'negative'} "
                    f"({prev['polarity']}) but Day {curr['day']} was "
                    f"{'positive' if curr['polarity'] > 0 else 'negative'} "
                    f"({curr['polarity']})"
                ),
            })
    return polarities, flags

# ── 6. MERGE INTO COHERENT ANSWER ─────────────────────────────
def build_answer(top_chunks, keyword):
    parts = []
    for c in top_chunks[:3]:
        sentences = [s.strip() for s in c["text"].split(".")
                     if keyword.lower() in s.lower() and len(s.strip()) > 10]
        if sentences:
            parts.append(f"(Day {c['day']}) {sentences[0][:150]}")
    if not parts:
        return f"No specific mentions of '{keyword}' found in top results."
    return " | ".join(parts)

# ── 7. MAIN RESOLVER ──────────────────────────────────────────
def resolve(query: str) -> dict:
    keyword = query.split()[-1].rstrip("?").lower()

    hits   = retrieve(query, top_k=15)
    ranked = rank_chunks(hits)[:5]

    polarities, flags = detect_contradictions(ranked, keyword)
    answer            = build_answer(ranked, keyword)

    return {
        "query":                query,
        "keyword_searched":     keyword,
        "chunks_retrieved":     len(hits),
        "answer":               answer,
        "contradictions_found": len(flags) > 0,
        "contradiction_count":  len(flags),
        "contradiction_details": flags,
        "top_chunks_ranked": [
            {
                "day":              r["day"],
                "final_score":      r["final_score"],
                "recency":          round(r["day"] / max_day, 3),
                "emotional_weight": r["emotional_weight"],
                "similarity":       round(r["similarity"], 4),
            }
            for r in ranked
        ],
        "sentiment_per_chunk": polarities,
    }

# ── 8. RUN TEST QUERIES ───────────────────────────────────────
print("\n====== RAG CONFLICT RESOLVER TEST ======\n")

queries = [
    "Did I mention anything about my sister?",
    "What did I say about work?",
    "Did I talk about my friend?",
]

for q in queries:
    print(f"QUERY: {q}")
    result = resolve(q)

    print(f"  Chunks retrieved : {result['chunks_retrieved']}")
    print(f"  Contradictions   : {result['contradiction_count']} found")

    if result["contradictions_found"]:
        for f in result["contradiction_details"]:
            print(f"    !! {f['detail']}")

    print(f"  Answer           : {result['answer'][:200]}")
    print(f"  Top chunks ranked:")
    for c in result["top_chunks_ranked"]:
        print(f"    Day {c['day']:>2} | score={c['final_score']} | "
              f"sim={c['similarity']} | emotion={c['emotional_weight']:.1f}")
    print()

print("DONE - Part 3 complete")