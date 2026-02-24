# voc-insight-hub
VoC Insight Hub
# VoC Insight Hub — From Customer Feedback to Prioritized Product Themes (Streamlit)

Live app: https://voc-insight-app.streamlit.app/
Repo: https://github.com/rishav98sin-max/voc-insight-hub

VoC Insight Hub is a Product Insights mini-system that converts raw customer feedback into **prioritized** themes and recommended next actions. It’s designed to help a PM quickly answer:
- What are the top problems customers are reporting?
- Which themes are most urgent and most valuable to address?
- What should the team do next, and how will we measure success?

This project demonstrates end-to-end Product Thinking: turning unstructured VoC into an actionable, measurable, stakeholder-ready output (CSV + PowerPoint).

---

## Why this matters 
In real teams, feedback lives across support tickets, surveys, reviews, and sales notes. The challenge isn’t “getting feedback”—it’s synthesizing it into:
- Clear themes
- A prioritization rationale
- A next-step plan with ownership and success metrics

VoC Insight Hub solves that by clustering similar feedback, summarizing each theme, and ranking themes using a transparent scoring model.

---

## Key outcomes (what the app produces)
For each theme, the app generates a consistent “theme card”:
- Theme name
- 2‑sentence problem summary
- Sentiment (positive/neutral/negative)
- Opportunity statement
- Recommended next step (action verb)
- Success metric (Activation, Retention, CSAT, etc.)
- Suggested owner (PM / Eng / Design / Support)
- Confidence level (low/medium/high)
- Evidence quotes (representative feedback snippets)

It also exports:
- `themes.csv` (theme-level view)
- `clustered_feedback.csv` (feedback with cluster ids)
- `voc_insights_summary.pptx` (stakeholder-ready summary deck)

---

## Prioritization model (explainable by design)
Each theme receives a `priority_score`:

priority_score = count × severity_component × avg_plan_weight

Where:
- count = number of feedback items in the theme (frequency / volume)
- severity_component = average severity score if provided (low=1, medium=2, high=3, critical=4), default baseline 1.5 if missing
- avg_plan_weight = average customer value weighting (free=1.0, pro=1.5, team=2.0, enterprise=2.5), default 1.0 if missing

This is intentionally simple:
- Easy to explain to stakeholders
- Easy to tweak based on business context
- Avoids “black box prioritization”

---

## How it works (high-level flow)
1) Upload feedback CSV (requires `text` column)
2) Optional filters (product, persona, plan, severity, etc.)
3) Convert text → embeddings
4) Cluster embeddings into K themes (KMeans)
5) For each theme:
   - compute priority inputs
   - generate a structured theme card via Groq (JSON schema)
6) Review results in tables + drill down to evidence
7) Export CSVs + PPT

---

## Demo dataset
This repo includes `sample_feedback.csv` (root).  
In the app, click **Download sample_feedback.csv**, upload it, then click **Generate insights**.

---

## How to run locally
### Install
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
