import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("YouTube Sentiment Analyzer")
st.caption("Move comments between Negative / Neutral / Positive with one click. Metadata included (likes, replies, date).")

API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", "./results/checkpoint-2031")
LABELS = ["negative", "neutral", "positive"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for k, v in {
    "df": None,
    "last_url": "",
    "video_id": None,
    "loaded": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        os.path.abspath(MODEL_PATH),
        local_files_only=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model

if not API_KEY:
    st.error("Missing YOUTUBE_API_KEY. Copy .env.example to .env and add your YouTube Data API key.")
    st.stop()

if not os.path.isdir(MODEL_PATH):
    st.error(
        f"Model checkpoint not found at {MODEL_PATH!r}. "
        "Place the fine-tuned checkpoint there or set MODEL_PATH in your environment."
    )
    st.stop()

try:
    tokenizer, model = load_model()
except Exception as exc:
    st.error(f"Could not load the model checkpoint: {exc}")
    st.stop()

def extract_video_id(url: str):
    url = (url or "").strip()
    if not url:
        return None
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    return None

def parse_dt(x):
    if not x:
        return pd.NaT
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return pd.NaT

def fetch_comments_with_meta(video_id: str, max_results: int = 500):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    out = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results, 100),
        textFormat="plainText",
        order="relevance",
    )

    while request is not None and len(out) < max_results:
        try:
            response = request.execute()
        except HttpError as e:
            st.error(f"YouTube API error: {e}")
            break

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            sn = top["snippet"]
            text = sn.get("textDisplay", "")
            if not text:
                continue

            out.append({
                "comment_id": top.get("id", ""),
                "comment": text,
                "likes": int(sn.get("likeCount", 0)),
                "replies": int(item["snippet"].get("totalReplyCount", 0)),
                "publishedAt": sn.get("publishedAt", None),
                "updatedAt": sn.get("updatedAt", None),
            })

            if len(out) >= max_results:
                break

        request = youtube.commentThreads().list_next(request, response)

    return out[:max_results]

def predict_batch(texts, max_length=128, batch_size=32):
    if not texts:
        return [], [], np.zeros((0, 3), dtype=np.float32)

    all_preds, all_confs = [], []
    probs_chunks = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        pred_ids = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        probs_chunks.append(probs)
        all_preds.extend([LABELS[j] for j in pred_ids])
        all_confs.extend(confs.tolist())

    return all_preds, all_confs, np.vstack(probs_chunks)

def apply_filters(df, search, min_likes, min_replies, conf_max):
    out = df.copy()
    out = out[out["confidence"] <= conf_max]
    out = out[out["likes"] >= min_likes]
    out = out[out["replies"] >= min_replies]
    if search:
        s = search.lower().strip()
        out = out[out["comment"].str.lower().str.contains(s, na=False)]
    return out

def apply_sort(df, sort_by):
    out = df.copy()
    if sort_by == "date_newest":
        out = out.sort_values("published_dt", ascending=False)
    elif sort_by == "date_oldest":
        out = out.sort_values("published_dt", ascending=True)
    elif sort_by == "likes_desc":
        out = out.sort_values("likes", ascending=False)
    elif sort_by == "replies_desc":
        out = out.sort_values("replies", ascending=False)
    elif sort_by == "confidence_asc":
        out = out.sort_values("confidence", ascending=True)
    elif sort_by == "confidence_desc":
        out = out.sort_values("confidence", ascending=False)
    return out

with st.sidebar:
    st.header("Settings")
    max_comments = st.slider("Max comments to fetch", 50, 2000, 500, 50)
    max_length = st.select_slider("Tokenizer max_length", [64, 96, 128, 160, 192], value=128)
    batch_size = st.select_slider("Batch size", [8, 16, 32, 64], value=32)

    st.divider()
    st.subheader("Filter & Sort")
    search = st.text_input("Search text contains", value="")
    min_likes = st.number_input("Min likes", min_value=0, value=0, step=1)
    min_replies = st.number_input("Min replies", min_value=0, value=0, step=1)
    conf_max = st.slider("Show confidence ≤", 0.0, 1.0, 1.0, 0.01)
    sort_by = st.selectbox(
        "Sort by",
        ["date_newest", "date_oldest", "likes_desc", "replies_desc", "confidence_asc", "confidence_desc"],
        index=0
    )

    st.divider()
    per_col_limit = st.slider("Max cards shown per column", 50, 800, 250, 50)

    st.divider()
    if st.button("Clear loaded video/results"):
        st.session_state.df = None
        st.session_state.last_url = ""
        st.session_state.video_id = None
        st.session_state.loaded = False
        for k in list(st.session_state.keys()):
            if k.startswith("move_"):
                del st.session_state[k]
        st.success("Cleared.")

video_url = st.text_input(
    "YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
    value=st.session_state.last_url
)

colA, colB = st.columns([1, 1])
with colA:
    analyze_btn = st.button("Analyze Video", type="primary")
with colB:
    st.caption(f"Device: **{device.type.upper()}**")

if analyze_btn:
    vid = extract_video_id(video_url)
    if not vid:
        st.error("Invalid YouTube URL.")
        st.stop()

    st.session_state.last_url = video_url
    st.session_state.video_id = vid

    with st.spinner("Fetching comments (likes/replies/date)..."):
        items = fetch_comments_with_meta(vid, max_results=max_comments)

    if not items:
        st.warning("No comments fetched.")
        st.stop()

    texts = [x["comment"] for x in items]

    with st.spinner("Analyzing sentiment (batch inference)..."):
        t0 = time.time()
        preds, confs, probs = predict_batch(texts, max_length=max_length, batch_size=batch_size)
        dt = time.time() - t0

    df = pd.DataFrame(items)
    df["predicted"] = preds
    df["confidence"] = confs
    df["p_negative"] = probs[:, 0]
    df["p_neutral"] = probs[:, 1]
    df["p_positive"] = probs[:, 2]
    df["label"] = df["predicted"]

    df["published_dt"] = df["publishedAt"].apply(parse_dt)

    df["card_id"] = df["comment_id"].fillna("").astype(str)
    df.loc[df["card_id"] == "", "card_id"] = df.index.map(lambda i: f"row_{i}")

    st.session_state.df = df
    st.session_state.loaded = True

    for k in list(st.session_state.keys()):
        if k.startswith("move_"):
            del st.session_state[k]

    st.success(f"Done! {len(df)} comments analyzed in {dt:.2f}s.")

df = st.session_state.df
if df is None:
    st.info("Paste a YouTube URL and click **Analyze Video**.")
    st.stop()

for cid in df["card_id"].tolist():
    key = f"move_{cid}"
    if key in st.session_state:
        df.loc[df["card_id"] == cid, "label"] = st.session_state[key]
st.session_state.df = df

df_view = apply_filters(df, search, min_likes, min_replies, conf_max)
df_view = apply_sort(df_view, sort_by)

st.subheader("Summary")
counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1: st.metric("Negative", int(counts["negative"]))
with c2: st.metric("Neutral", int(counts["neutral"]))
with c3: st.metric("Positive", int(counts["positive"]))
with c4: st.metric("Changed vs predicted", int((df["label"] != df["predicted"]).sum()))

st.subheader("Kanban Review")

for cid in df["card_id"].tolist():
    key = f"move_{cid}"
    if key in st.session_state:
        df.loc[df["card_id"] == cid, "label"] = st.session_state[key]

st.session_state.df = df

df_view = apply_filters(df, search, min_likes, min_replies, conf_max)
df_view = apply_sort(df_view, sort_by)

colN, colU, colP = st.columns(3)

def render_column(lab, column):
    with column:
        st.markdown(f"### {lab.upper()} ({(df['label']==lab).sum()})")

        subset = df_view[df_view["label"] == lab].head(per_col_limit)

        if subset.empty:
            st.caption("No comments in this group.")
            return

        for _, row in subset.iterrows():
            cid = row["card_id"]

            with st.container(border=True):
                st.write(row["comment"])

                st.caption(
                    f"👍 {row['likes']}   "
                    f"💬 {row['replies']}   "
                    f"📅 {row['published_dt'].strftime('%Y-%m-%d') if pd.notnull(row['published_dt']) else 'unknown'}   "
                    f"conf {row['confidence']:.3f}"
                )

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("Neg", key=f"{cid}_neg"):
                        st.session_state[f"move_{cid}"] = "negative"
                        st.rerun()
                with b2:
                    if st.button("Neu", key=f"{cid}_neu"):
                        st.session_state[f"move_{cid}"] = "neutral"
                        st.rerun()
                with b3:
                    if st.button("Pos", key=f"{cid}_pos"):
                        st.session_state[f"move_{cid}"] = "positive"
                        st.rerun()

render_column("negative", colN)
render_column("neutral", colU)
render_column("positive", colP)

st.subheader("Download CSVs")

minimal_df = df[["comment", "label"]].rename(columns={"label": "sentiment"})
minimal_csv = minimal_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download MINIMAL CSV (comment, sentiment)",
    data=minimal_csv,
    file_name=f"youtube_comments_minimal_{st.session_state.video_id}.csv",
    mime="text/csv"
)

full_cols = [
    "comment_id", "comment",
    "predicted", "label", "confidence",
    "p_negative", "p_neutral", "p_positive",
    "likes", "replies", "publishedAt", "updatedAt"
]
full_df = df[full_cols].copy()
full_csv = full_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download FULL CSV (all parameters)",
    data=full_csv,
    file_name=f"youtube_comments_full_{st.session_state.video_id}.csv",
    mime="text/csv"
)
