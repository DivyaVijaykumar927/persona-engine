import pandas as pd
import joblib
import time
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# ── 1. TRAINING DATA ──────────────────────────────────────────
training_data = [
    # reminder
    ("remind me to call mom tomorrow", "reminder"),
    ("don't let me forget the meeting at 3pm", "reminder"),
    ("set a reminder for my dentist appointment", "reminder"),
    ("alert me when the timer goes off", "reminder"),
    ("can you remind me about this later", "reminder"),
    ("I need to remember to pay the bill", "reminder"),
    ("please remind me at 6pm", "reminder"),
    ("don't forget to pick up groceries", "reminder"),
    ("reminder for my sister's birthday", "reminder"),
    ("set an alarm for tomorrow morning", "reminder"),
    ("notify me before the meeting starts", "reminder"),
    ("I keep forgetting to take my medicine", "reminder"),

    # emotional-support
    ("I feel so alone lately", "emotional-support"),
    ("I am really stressed about work", "emotional-support"),
    ("nobody understands me", "emotional-support"),
    ("I am going through a tough time", "emotional-support"),
    ("I feel anxious and overwhelmed", "emotional-support"),
    ("I just need someone to talk to", "emotional-support"),
    ("I am feeling really down today", "emotional-support"),
    ("everything feels hopeless right now", "emotional-support"),
    ("I had a really bad day", "emotional-support"),
    ("I am so frustrated with my life", "emotional-support"),
    ("I feel like crying for no reason", "emotional-support"),
    ("my anxiety is really bad today", "emotional-support"),
    ("I am scared about the future", "emotional-support"),
    ("I feel like a failure", "emotional-support"),

    # action-item
    ("can you write an email to my boss", "action-item"),
    ("search for the best restaurants nearby", "action-item"),
    ("make a list of things to buy", "action-item"),
    ("calculate the total for me", "action-item"),
    ("book a flight to Mumbai", "action-item"),
    ("find me a recipe for pasta", "action-item"),
    ("translate this sentence to French", "action-item"),
    ("summarize this document", "action-item"),
    ("create a schedule for next week", "action-item"),
    ("order food from Swiggy", "action-item"),
    ("send a message to John", "action-item"),
    ("look up the weather for tomorrow", "action-item"),
    ("make a todo list for the project", "action-item"),
    ("draft a reply to this email", "action-item"),

    # small-talk
    ("hey how are you doing", "small-talk"),
    ("what is your favorite color", "small-talk"),
    ("nice weather today huh", "small-talk"),
    ("tell me a joke", "small-talk"),
    ("what do you think about movies", "small-talk"),
    ("do you like music", "small-talk"),
    ("hi there how is it going", "small-talk"),
    ("good morning hope you have a great day", "small-talk"),
    ("what did you do today", "small-talk"),
    ("lets just chat for a bit", "small-talk"),
    ("I am bored what should we talk about", "small-talk"),
    ("do you have any hobbies", "small-talk"),
    ("what is your name", "small-talk"),
    ("can we just talk", "small-talk"),

    # unknown
    ("xkcd 1234 flux capacitor override", "unknown"),
    ("zzzz blah blah nonsense", "unknown"),
    ("asdfgh qwerty random stuff", "unknown"),
    ("nothing just testing", "unknown"),
]

# ── 2. ADD SMALL-TALK FROM YOUR CSV ───────────────────────────
print("Loading extra small-talk from conversations.csv ...")
try:
    df = pd.read_csv("data/conversations.csv", header=None, names=["conv"])
    for _, row in df.sample(min(300, len(df)), random_state=42).iterrows():
        lines = [l.replace("User 1:", "").strip()
                 for l in str(row["conv"]).split("\n")
                 if l.startswith("User 1:")]
        for line in lines[:2]:
            if line and len(line) > 5:
                training_data.append((line, "small-talk"))
    print(f"  Added CSV samples. Total training examples: {len(training_data)}")
except Exception as e:
    print(f"  Could not load CSV: {e}. Continuing with base data.")

# ── 3. TRAIN MODEL ────────────────────────────────────────────
texts, labels = zip(*training_data)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf",   LinearSVC(max_iter=2000, C=1.0))
])

print("\nTraining model ...")
model.fit(texts, labels)

# ── 4. VALIDATE ───────────────────────────────────────────────
scores = cross_val_score(model, texts, labels, cv=3)
print(f"Cross-validation accuracy: {scores.mean():.2%}")

# ── 5. TEST SPEED ─────────────────────────────────────────────
start = time.time()
for _ in range(200):
    model.predict(["remind me to call mom"])
elapsed = (time.time() - start) / 200 * 1000
print(f"Average inference time: {elapsed:.2f} ms  (limit: 200ms)")

# ── 6. CHECK SIZE ─────────────────────────────────────────────
joblib.dump(model, "intent_classifier.pkl")
size_mb = os.path.getsize("intent_classifier.pkl") / 1_000_000
print(f"Model size: {size_mb:.3f} MB  (limit: 50MB)")

# ── 7. CLASSIFY FUNCTION ──────────────────────────────────────
def classify_intent(message: str) -> dict:
    start = time.time()
    pred       = model.predict([message])[0]
    scores_raw = model.decision_function([message])[0]
    confidence = float(max(scores_raw))
    elapsed_ms = (time.time() - start) * 1000

    if confidence < 0.2:
        pred = "unknown"

    return {
        "message":    message,
        "intent":     pred,
        "confidence": round(confidence, 3),
        "latency_ms": round(elapsed_ms, 2),
    }

# ── 8. TEST IT ────────────────────────────────────────────────
print("\n====== INTENT CLASSIFIER TEST ======\n")
test_messages = [
    "remind me to buy groceries tomorrow",
    "I feel really sad and lonely today",
    "can you search for flights to Delhi",
    "hey what is up how are you",
    "set an alarm for 7am",
    "I am so stressed about my exams",
    "make a list of things I need to pack",
    "good morning friend",
    "qwerty blah nonsense xyz",
]

for msg in test_messages:
    result = classify_intent(msg)
    print(f"  [{result['intent']:<18}] {result['confidence']:>6.2f} conf | {result['latency_ms']:.1f}ms | \"{msg}\"")

print("\nDONE - Model saved as intent_classifier.pkl")