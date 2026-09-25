"""
Streamlit UI for the semantic music search system.

Usage:
    streamlit run app.py
"""
import html

import pandas as pd
import streamlit as st

from retrieval import BASE_DIR, load_index, load_model
from search import search

RAW_DATA_PATH = BASE_DIR / "data" / "spotify_songs.csv"
EXAMPLES = [
    "songs for a late-night drive",
    "happy energetic songs for a workout",
    "sad acoustic breakup song",
    "chill r&b for late night",
    "party reggaeton",
]

st.set_page_config(page_title="Semantic Music Search", page_icon="🎧", layout="centered")

st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 3rem; max-width: 760px;}
      .hero h1 {font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
      .hero p {opacity: 0.65; margin-top: 0;}
      .card {display: flex; gap: 1rem; align-items: center; padding: 0.85rem 1rem;
             border: 1px solid rgba(128,128,128,0.18); border-radius: 12px; margin-bottom: 0.6rem;}
      .rank {font-size: 1.1rem; font-weight: 600; opacity: 0.35; width: 1.8rem; text-align: right;}
      .body {flex: 1; min-width: 0;}
      .title {font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
      .title a {color: inherit; text-decoration: none;}
      .title a:hover {text-decoration: underline;}
      .sub {font-size: 0.87rem; opacity: 0.65; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
      .tag {display: inline-block; font-size: 0.72rem; padding: 0.1rem 0.5rem; border-radius: 999px;
            background: rgba(29,185,84,0.14); color: #1DB954; margin-right: 0.3rem; margin-top: 0.3rem;}
      .score {text-align: right; font-size: 0.75rem; opacity: 0.6; width: 5.5rem;}
      .bar {height: 4px; border-radius: 2px; background: rgba(128,128,128,0.18); margin-top: 0.3rem;}
      .bar > div {height: 100%; border-radius: 2px; background: #1DB954;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Loading song index…")
def get_index():
    return load_index()


@st.cache_data(show_spinner=False)
def get_details():
    """Extra display metadata (album, year, genre) from the raw catalog, if present."""
    if not RAW_DATA_PATH.exists():
        return None
    cols = ["track_id", "track_album_name", "track_album_release_date", "playlist_genre", "playlist_subgenre"]
    return pd.read_csv(RAW_DATA_PATH, usecols=cols).drop_duplicates("track_id").set_index("track_id")


def use_example():
    if st.session_state.example:
        st.session_state.query = st.session_state.example
    st.session_state.example = None


def render_card(rank, row, details):
    e = html.escape
    url = f"https://open.spotify.com/track/{e(str(row['track_id']))}"
    sub = e(row["track_artist"])
    tags = ""
    if details is not None and row["track_id"] in details.index:
        d = details.loc[row["track_id"]]
        year = str(d["track_album_release_date"])[:4]
        sub += f" · {e(str(d['track_album_name']))} · {e(year)}"
        tags = "".join(
            f'<span class="tag">{e(str(t))}</span>' for t in (d["playlist_genre"], d["playlist_subgenre"]) if pd.notna(t)
        )
    match = max(0.0, min(1.0, float(row["similarity"]))) * 100
    st.markdown(
        f"""
        <div class="card">
          <div class="rank">{rank}</div>
          <div class="body">
            <div class="title"><a href="{url}" target="_blank">{e(row['track_name'])}</a></div>
            <div class="sub">{sub}</div>
            <div>{tags}</div>
          </div>
          <div class="score">
            match {row['similarity']:.2f}<div class="bar"><div style="width:{match:.0f}%"></div></div>
            <div style="margin-top:0.35rem">popularity {int(row['track_popularity'])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><h1>🎧 Semantic Music Search</h1>'
    "<p>Describe a mood, moment or vibe — get songs that match its meaning, not just its keywords.</p></div>",
    unsafe_allow_html=True,
)

if "query" not in st.session_state:
    st.session_state.query = st.query_params.get("q", "")  # allows shareable links: ?q=...

query = st.text_input(
    "Search", key="query", placeholder="e.g. songs for a late-night drive", label_visibility="collapsed"
)

st.pills("Try", EXAMPLES, key="example", on_change=use_example, label_visibility="collapsed")

with st.expander("Settings"):
    top_k = st.slider("Number of results", 1, 50, 10)
    hybrid = st.toggle("Boost popular songs (hybrid ranking)", value=True)
    alpha = st.slider(
        "Semantic weight (alpha)", 0.0, 1.0, 0.8, 0.05, disabled=not hybrid,
        help="1.0 = pure semantic similarity, 0.0 = pure popularity.",
    )

if query.strip():
    try:
        index, model = get_index(), get_model()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    with st.spinner("Searching…"):
        results = search(query, top_k=top_k, alpha=alpha, hybrid=hybrid, index=index, model=model)
    details = get_details()
    st.caption(f"Top {len(results)} results for “{query.strip()}”")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_card(rank, row, details)
