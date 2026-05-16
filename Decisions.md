Design Decisions
Why TF-IDF + LinearSVC instead of a Transformer model?
The brief was very clear — under 50MB, under 200ms, runs fully on CPU.
If I had used DistilBERT or any HuggingFace transformer model, the size alone
would be 250MB — five times over the limit — and inference would take 400 to
800ms on CPU.
My approach gives a model that is under 1MB and runs in under 5ms.
That is a 50x smaller model and 100x faster — with perfectly acceptable
accuracy for this use case.
TF-IDF + LinearSVC (my choice)DistilBERT (transformer)Model size< 1 MB~250 MBInference speed< 5 ms400–800 msRuns offlineYesYesAPI calls neededNoneNoneMeets brief constraintsYESNO

Why does recency get the highest weight in RAG ranking?
I weighted recency at 40%, higher than similarity at 25%.
The reason is simple — in a personal conversation system, what someone said
last week is almost always more relevant than what they said two months ago,
even if the older message has more matching keywords.
Recency reflects how human memory actually works. The most recent emotional
context is the most current truth about that person.
Ranking formula used:
Final Score = (0.40 x recency) + (0.35 x emotional weight) + (0.25 x similarity)
WeightFactorWhy40%RecencyRecent context is more relevant to current state35%Emotional weightEmotionally charged messages are more memorable and significant25%TF-IDF similarityKeyword match still matters but is not the only signal

Why keep raw messages local and not sync them?
Raw conversation data contains sensitive personal and emotional information.
Syncing it to cloud storage creates privacy and security risks.
Only derived, non-sensitive data is synced:

Persona drift timeline
Intent model weights
RAG index snapshots

Raw messages and emotional context stay on-device only. This is the minimum
data needed for cross-device continuity without compromising user privacy.

Summary
These two decisions together — a lightweight offline classifier and a
recency-first retrieval system — make this a practical, privacy-respecting,
real-world deployable system.
