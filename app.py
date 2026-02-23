import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

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
    "required"
