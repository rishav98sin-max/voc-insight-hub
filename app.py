import hashlib
import io
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ----------------------------
# Config
# ----------------------------
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"  # Groq OpenAI-compatible endpoint [web:17]
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="VoC Theme Finder", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> SentenceTransformer:
    # SentenceTransformer encode supports show_progress_bar param [web:36]
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = load_embedder(model_name)
    emb = model.encode(
        texts,
        show_progress_bar=False,  # keep server logs clean; UI has its own status [web:36]
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)

def groq_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    temperature: float = 0.2,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,  # structured outputs [web:17]
    }
    r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def groq_generate_theme_json(
    api_key: str,
    model: str,
    cluster_examples: List[str],
) -> Dict[str, Any]:
    schema = {
        "name": "theme_label",
        "strict": False,  # best-effort mode; default is false [web:17]
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "theme": {"type": "string"},
                "one_liner": {"type": "string"},
                "why_it_matters": {"type": "string"},
                "success_metric": {"type": "string"},
                "owner": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": [
                "theme",
                "one_liner",
                "why_it_matters",
                "success_metric",
                "owner",
                "confidence",
            ],
        },
    }

    examples = "\n".join([f"- {t}" for t in cluster_examples[:12]])

    system = (
        "You are a product analyst. You will label a cluster of customer feedback items with a concise theme "
        "and provide short, practical fields for a PM team. Output must be JSON only."
    )
    user = (
        "Cluster examples:\n"
        f"{examples}\n\n"
        "Create a theme that captures the common issue/request.\n"
        "Keep all fields short and business-friendly."
    )

    data = groq_chat(
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": schema},  # [web:17]
        temperature=0.2,
        timeout_s=90,
    )

    content = data["choices"][0]["message"]["content"]
    # content should be JSON text; parse it
    return json.loads(content)

def safe_theme_fallback() -> Dict[str, Any]:
    return {
        "theme": "Unlabeled theme (fallback)",
        "one_liner": "Could not label this cluster due to an API/format issue.",
        "why_it_matters": "Manual review needed to avoid missing a recurring customer pain point.",
        "success_metric": "Reduction in related complaints",
        "owner": "PM",
        "confidence": "low",
    }

# ----------------------------
# UI
# ----------------------------
st.title("VoC Theme Finder")
st.caption("Upload feedback → cluster similar items → label each cluster with Groq")

with st.sidebar:
    st.header("Settings")

    groq_api_key = st.text_input("GROQ_API_KEY", type="password")
    groq_model = st.text_input("Groq model", value="llama-3.1-8b-instant")

    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)

    k = st.slider("Number of themes (k)", min_value=2, max_value=12, value=5)
    max_rows = st.slider("Max rows to process", min_value=50, max_value=1000, value=200, step=50)
    throttle_s = st.slider("Throttle between Groq calls (seconds)", 0.0, 2.0, 0.5, 0.1)

    st.divider()
    st.write("Input format:")
    st.code("CSV with a column named: feedback", language="text")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
run = st.button("Generate insights", type="primary", use_container_width=True)

if not uploaded:
    st.info("Upload a CSV first (needs a `feedback` column).")
    st.stop()

file_bytes = uploaded.getvalue()
file_sig = _sha1_bytes(file_bytes)

df = pd.read_csv(io.BytesIO(file_bytes))
if "feedback" not in df.columns:
    st.error("Your CSV must contain a column named `feedback`.")
    st.stop()

df = df.copy()
df["feedback"] = df["feedback"].astype(str).fillna("").str.strip()
df = df[df["feedback"].str.len() > 0].head(max_rows).reset_index(drop=True)

st.write(f"Rows loaded: {len(df)}")

if not run:
    st.stop()

if not groq_api_key:
    st.error("Please enter GROQ_API_KEY in the sidebar.")
    st.stop()

# ----------------------------
# Step 1/3: Embeddings + Clustering
# ----------------------------
st.info("Step 1/3: Starting embedding + clustering...")

with st.spinner("Embedding feedback and clustering..."):
    texts = df["feedback"].tolist()
    X = embed_texts(texts, embed_model)

    k_eff = min(k, len(df)) if len(df) > 0 else k
    if k_eff < 2:
        st.error("Need at least 2 rows to cluster.")
        st.stop()

    km = KMeans(n_clusters=k_eff, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    df["cluster"] = labels

st.success("Step 1/3 complete: clustering done.")
st.caption(f"Run settings: k={k_eff}, rows={len(df)}, embed_model={embed_model}, groq_model={groq_model}")

# ----------------------------
# Step 2/3: Preview clusters
# ----------------------------
st.info("Step 2/3: Preview clusters")

cluster_counts = df["cluster"].value_counts().sort_index()
st.write("Cluster sizes:")
st.dataframe(cluster_counts.rename("count").reset_index().rename(columns={"index": "cluster"}), use_container_width=True)

with st.expander("Show sample rows per cluster"):
    for c in sorted(df["cluster"].unique()):
        st.markdown(f"### Cluster {c} ({int(cluster_counts.loc[c])} items)")
        st.write(df[df["cluster"] == c]["feedback"].head(5).tolist())

# ----------------------------
# Step 3/3: Groq labeling
# ----------------------------
st.info("Step 3/3: Starting Groq labeling...")

themes: List[Dict[str, Any]] = []
clusters = sorted(df["cluster"].unique())

status = st.empty()
progress = st.progress(0, text="Preparing to label clusters...")  # progress bar API [web:32]

for i, c in enumerate(clusters, start=1):
    status.info(f"Labeling theme {i}/{len(clusters)} (cluster {c})...")
    progress.progress(int((i - 1) / max(1, len(clusters)) * 100), text=f"Labeling {i}/{len(clusters)}...")  # [web:32]

    cluster_texts = df[df["cluster"] == c]["feedback"].tolist()

    try:
        obj = groq_generate_theme_json(
            api_key=groq_api_key,
            model=groq_model,
            cluster_examples=cluster_texts,
        )
    except Exception as e:
        st.error(f"Groq labeling failed for cluster {c}: {e}")
        obj = safe_theme_fallback()

    obj["cluster"] = int(c)
    obj["n_items"] = int(len(cluster_texts))
    themes.append(obj)

    if throttle_s and throttle_s > 0:
        time.sleep(float(throttle_s))

progress.progress(100, text="Done labeling.")
status.success("All clusters labeled.")

themes_df = pd.DataFrame(themes).sort_values("cluster").reset_index(drop=True)

st.subheader("Themes")
st.dataframe(themes_df, use_container_width=True)

# Join themes back to rows
df_out = df.merge(themes_df[["cluster", "theme"]], on="cluster", how="left")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download themes.csv",
        data=df_to_csv_bytes(themes_df),
        file_name="themes.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    st.download_button(
        "Download clustered_feedback.csv",
        data=df_to_csv_bytes(df_out),
        file_name="clustered_feedback.csv",
        mime="text/csv",
        use_container_width=True,
    )
