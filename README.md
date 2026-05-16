# Persona Engine

AI system that tracks user mood drift, classifies intent offline, and resolves RAG conflicts.

## Live Demo
[Click here](YOUR_STREAMLIT_LINK_HERE)

## Parts
- Part 1: Persona drift detector — tracks tone changes across days
- Part 2: Offline intent classifier — TF-IDF + LinearSVC, under 1MB, under 5ms
- Part 3: RAG conflict resolver — ranks by recency + emotional weight, flags contradictions

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## System Design
- On-device: SQLite stores messages, persona snapshots, RAG chunks
- Syncs: persona timeline + intent model to cloud
- Stays local: raw messages + emotional context (privacy)
- Conflict resolution: last-write-wins with timestamp

## Architecture
Device (SQLite) → Sync Layer (FastAPI) → Cloud Storage (S3)