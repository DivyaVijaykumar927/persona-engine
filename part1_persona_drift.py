import pandas as pd
import re
from textblob import TextBlob

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("data/conversations.csv", header=None, names=["conversation"])

# Simulate days: every 1000 rows = 1 day
df["day"] = (df.index // 1000) + 1

# ── 2. EXTRACT USER 1 MESSAGES ────────────────────────────────
def extract_user1(text):
    lines = str(text).split("\n")
    msgs = [l.replace("User 1:", "").strip()
            for l in lines if l.startswith("User 1:")]
    return " ".join(msgs)

df["user_text"] = df["conversation"].apply(extract_user1)

# ── 3. DETECT TONE ────────────────────────────────────────────
def detect_tone(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity  # -1 to +1

    text_lower = text.lower()
    words = text_lower.split()

    # Count keyword hits (need 2+ hits to qualify, not just 1)
    frustrated_kw = ["why", "ugh", "again", "still", "not working",
                     "can't", "won't", "terrible", "awful", "hate", "annoying"]
    playful_kw    = ["haha", "lol", "hehe", "hilarious", "joke",
                     "funny", "laugh", "playful", "silly", "tease"]
    formal_kw     = ["therefore", "however", "regarding", "sincerely",
                     "furthermore", "accordingly", "pursuant", "hence"]
    curious_kw    = ["how", "what", "why", "when", "could you",
                     "tell me", "explain", "wondering", "curious", "?"]

    frustrated_hits = sum(1 for k in frustrated_kw if k in text_lower)
    playful_hits    = sum(1 for k in playful_kw    if k in text_lower)
    formal_hits     = sum(1 for k in formal_kw     if k in text_lower)
    curious_hits    = sum(1 for k in curious_kw    if k in text_lower)

    # Need at least 2 keyword hits to assign that tone
    if frustrated_hits >= 2 and polarity < 0:
        return "frustrated"
    if playful_hits >= 2:
        return "playful"
    if formal_hits >= 1:
        return "formal"
    if curious_hits >= 2:
        return "curious"
    # Polarity-based fallback
    if polarity > 0.3:
        return "positive"
    if polarity < -0.2:
        return "frustrated"
    return "casual"

# ── 4. DETECT TRIGGER ─────────────────────────────────────────
def detect_trigger(text):
    people = re.findall(r'\bmy\s+(\w+)\b', text, re.I)
    topics = ["work", "sister", "family", "boss", "project",
              "deadline", "school", "money", "health", "friend", "meeting"]

    if people:
        return f"person/relation: '{people[0]}'"
    found = [t for t in topics if t in text.lower()]
    if found:
        return f"topic: '{found[0]}'"
    return "unknown trigger"

# ── 5. BUILD TIMELINE ─────────────────────────────────────────
def build_timeline(df):
    # Get the most common tone per day
    day_tones = (df.groupby("day")["tone"]
                   .agg(lambda x: x.mode()[0])
                   .reset_index())

    timeline = []
    prev_tone = None

    for _, row in day_tones.iterrows():
        day  = row["day"]
        tone = row["tone"]

        entry = {
            "day":     day,
            "tone":    tone,
            "drift":   False,
            "trigger": None,
        }

        if prev_tone and tone != prev_tone:
            entry["drift"]   = True
            day_text         = " ".join(df[df["day"] == day]["user_text"])
            entry["trigger"] = detect_trigger(day_text)

        timeline.append(entry)
        prev_tone = tone

    return timeline

# ── 6. RUN ────────────────────────────────────────────────────
df["tone"] = df["user_text"].apply(detect_tone)
timeline   = build_timeline(df)

print("\n====== PERSONA DRIFT TIMELINE ======\n")
for t in timeline:
    if t["drift"]:
        print(f"  Day {t['day']:>2}  ->  {t['tone']:<12}  *** DRIFT detected  ({t['trigger']})")
    else:
        print(f"  Day {t['day']:>2}  ->  {t['tone']}")

print("\n====== TONE DISTRIBUTION ======\n")
print(df["tone"].value_counts().to_string())

# Also save results to a CSV for use in app.py
timeline_df = pd.DataFrame(timeline)
timeline_df.to_csv("data/persona_timeline.csv", index=False)
print("\nDONE - Saved timeline to data/persona_timeline.csv")