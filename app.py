import hashlib
import io
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"  # Groq OpenAI-compatible endpoint [web:631]
DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"

# Ollama structured outputs schema (also included in prompt for grounding) [web:307][web:295]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "theme_name": {"type": "string"},
        "problem_summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "opportunity": {"type": "string"},
        "next_step": {"type": "string"},
        "success_metric": {
            "type": "string",
            "enum": ["Activation rate", "Time-to-value", "Error rate", "Retention", "CSAT", "Support ticket rate"],
        },
        "owner": {"type": "string", "enum": ["PM", "Eng", "Design", "Support"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "theme_name",
        "problem_summary",
        "sentiment",
        "opportunity",
        "next_step",
        "success_metric",
        "owner",
        "confidence",
    ],
    "additionalProperties": False,
}

st.set_page_config(page_title="VoC Insight Agent", layout="wide")
st.title("VoC Insight Hub")

# ---------- Helpers ----------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def file_bytes_and_hash(uploaded_file):
    uploaded_file.seek(0)
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("Uploaded file is empty (0 bytes). Please re-upload.")
    h = hashlib.sha256(data).hexdigest()
    return data, h

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    data, _ = file_bytes_and_hash(uploaded_file)
    return pd.read_csv(io.BytesIO(data))

@st.cache_data(show_spinner=False)
def cached_embeddings(text_list: list[str], model_name: str, file_hash: str, max_rows: int) -> np.ndarray:
    embedder = load_embedder()
    emb = embedder.encode(text_list, show_progress_bar=False)
    return np.array(emb)

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def groq_generate_json(prompt: str, schema: dict, model: str) -> dict:
    api_key = st.secrets["GROQ_API_KEY"]  # Streamlit secrets [web:723]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON that matches the provided schema."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "voc_theme_schema",
                "strict": False,     # guaranteed schema compliance [web:709]
                "schema": schema,
            },
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=120)  # endpoint [web:631]
    r.raise_for_status()
    data = r.json()

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

@st.cache_data
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def severity_to_score(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(s, np.nan)

def plan_to_weight(x):
    if pd.isna(x):
        return 1.0
    s = str(x).strip().lower()
    return {"free": 1.0, "pro": 1.5, "team": 2.0, "enterprise": 2.5}.get(s, 1.0)

# ---------- UI controls ----------

uploaded = st.file_uploader("Upload feedback CSV (needs a 'text' column)", type=["csv"])
k = st.slider("Number of themes", 2, 12, 8)
max_rows = st.slider("Max rows to analyze (speed)", 50, 2000, 200)

if uploaded is None:
    st.info("Upload sample_feedback.csv")
    st.stop()

st.write("Uploaded file:", uploaded.name)

# Read df
try:
    df_raw = read_uploaded_csv(uploaded)
    _, file_hash = file_bytes_and_hash(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

if "text" not in df_raw.columns:
    st.error("CSV must contain a 'text' column.")
    st.stop()

# Optional filters
df = df_raw.copy()
with st.expander("Filters", expanded=True):
    for col in ["product", "persona", "plan", "source", "severity"]:
        if col in df.columns:
            options = sorted([x for x in df[col].dropna().unique()])
            chosen = st.multiselect(col.capitalize(), options)
            if chosen:
                df = df[df[col].isin(chosen)]

# Clean + limit
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str)
df = df.head(max_rows)

if len(df) == 0:
    st.warning("No rows left after filtering. Clear filters or upload more data.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

run = st.button("Generate insights")

# Show last run if available
if not run:
    if "themes_df" in st.session_state:
        st.subheader("Last run (saved in session)")
        st.dataframe(st.session_state["themes_df"], use_container_width=True)
    st.stop()

# ---------- Embedding + clustering ----------
with st.spinner("Embedding + clustering..."):
    texts = df["text"].tolist()
    emb = cached_embeddings(texts, "sentence-transformers/all-MiniLM-L6-v2", file_hash, max_rows)

    n_samples = emb.shape[0]
    if n_samples < 2:
        st.error("Not enough rows to cluster after filtering. Remove filters or increase max rows.")
        st.stop()

    k_eff = min(k, n_samples)
    if k_eff != k:
        st.warning(f"Reducing number of themes from {k} to {k_eff} because only {n_samples} rows are available.")

    km = KMeans(n_clusters=k_eff, n_init="auto", random_state=42)
    df["cluster"] = km.fit_predict(emb)

model = st.secrets.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
st.caption(f"Run settings: k={k_eff}, rows={len(df)}, model={model}")

# ---------- Theme scoring + labeling ----------
themes = []
progress = st.progress(0)

clusters = sorted(df["cluster"].unique())
total = len(clusters)

for idx, c in enumerate(clusters, start=1):
    dfc = df[df["cluster"] == c].copy()

    quotes = dfc["text"].head(5).tolist()
    evidence = " | ".join([q[:140].replace("\n", " ") for q in quotes])

    count = int(len(dfc))

    if "severity" in dfc.columns:
        dfc["_severity_score"] = dfc["severity"].apply(severity_to_score)
        avg_sev = float(np.nanmean(dfc["_severity_score"])) if dfc["_severity_score"].notna().any() else np.nan
    else:
        avg_sev = np.nan

    if "plan" in dfc.columns:
        dfc["_plan_weight"] = dfc["plan"].apply(plan_to_weight)
        avg_plan_weight = float(np.nanmean(dfc["_plan_weight"])) if dfc["_plan_weight"].notna().any() else 1.0
    else:
        avg_plan_weight = 1.0

    sev_component = (avg_sev if not np.isnan(avg_sev) else 1.5)
    priority_score = count * sev_component * avg_plan_weight

    prompt = f"""
You are a Product Manager analyzing customer feedback.

Return ONLY a JSON object with exactly these keys:
theme_name, problem_summary, sentiment, opportunity, next_step, success_metric, owner, confidence.

Rules:
- theme_name: 2-5 words, noun phrase, avoid "issues"
- problem_summary: exactly 2 sentences, do NOT copy snippets verbatim, no quotes
- sentiment: one of ["positive","neutral","negative"]
- opportunity: one sentence starting with "Opportunity:"
- next_step: one sentence, must start with an action verb (e.g., "Instrument", "Interview", "Reproduce", "A/B test", "Audit")
- owner: choose ONLY one from ["PM","Eng","Design","Support"] based on who would do the next_step
- success_metric: choose ONLY one from ["Activation rate","Time-to-value","Error rate","Retention","CSAT","Support ticket rate"]
- confidence: choose ONLY one from ["low","medium","high"] using this rubric:
  - high = snippets are specific and consistent (same problem repeated)
  - medium = theme is clear but details vary
  - low = ambiguous or mixed topics
Return ONLY JSON (no markdown, no extra text). [web:295]

Snippets:
{json.dumps(quotes, ensure_ascii=False)}
""".strip()

    with st.spinner(f"Labeling theme {idx}/{total} with Groq LLM..."):
        try:
            model = st.secrets.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            obj = groq_generate_json(prompt, OUTPUT_SCHEMA, model)
        except Exception as e:
            st.error(f"Groq call failed for cluster {c}: {e}")
            obj = {
            "theme_name": f"Theme {c}",
            "problem_summary": "Groq call failed. Theme label is a placeholder.",
            "sentiment": "neutral",
            "opportunity": "Opportunity: Retry with fewer rows/themes or increase timeout.",
            "next_step": "Retry labeling after checking Groq / rate limits.",
            "success_metric": "Activation rate",
            "owner": "PM",
            "confidence": "low",
        }

    themes.append({
        "cluster": int(c),
        "count": count,
        "avg_severity": avg_sev,
        "avg_plan_weight": avg_plan_weight,
        "priority_score": float(priority_score),
        "theme_name": obj.get("theme_name"),
        "problem_summary": obj.get("problem_summary"),
        "sentiment": obj.get("sentiment"),
        "opportunity": obj.get("opportunity"),
        "next_step": obj.get("next_step"),
        "success_metric": obj.get("success_metric"),
        "owner": obj.get("owner"),
        "confidence": obj.get("confidence"),
        "evidence_quotes": evidence,
    })

    progress.progress(int(idx / total * 100))

themes_df = pd.DataFrame(themes).sort_values("priority_score", ascending=False)

# Save to session_state for persistence across reruns [web:473]
st.session_state["themes_df"] = themes_df
st.session_state["clustered_df"] = df

# ---------- Outputs ----------
st.subheader("Top 5 themes (by priority)")
st.table(themes_df[["theme_name", "priority_score", "count", "owner", "success_metric", "confidence", "opportunity"]].head(5))

st.subheader("Themes (full)")
st.dataframe(
    themes_df[
        [
            "theme_name", "priority_score", "count",
            "owner", "success_metric", "confidence",
            "sentiment", "problem_summary", "opportunity", "next_step",
            "avg_severity", "avg_plan_weight", "evidence_quotes"
        ]
    ],
    use_container_width=True
)

st.subheader("Inspect a theme (drill-down)")
choice = st.selectbox("Theme", themes_df["theme_name"].tolist())
sel_cluster = int(themes_df.loc[themes_df["theme_name"] == choice, "cluster"].iloc[0])

cols_to_show = [c for c in ["feedback_id", "product", "date", "source", "persona", "plan", "severity", "text"] if c in df.columns]
if not cols_to_show:
    cols_to_show = ["text"]

st.dataframe(df[df["cluster"] == sel_cluster][cols_to_show].head(50), use_container_width=True)

st.download_button(
    "Download themes CSV",
    data=df_to_csv_bytes(themes_df),
    file_name="themes.csv",
    mime="text/csv",
)

df_out = df.copy()
st.download_button(
    "Download clustered feedback CSV",
    data=df_to_csv_bytes(df_out),
    file_name="clustered_feedback.csv",
    mime="text/csv",
)






