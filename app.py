"""Streamlit interface for the plot based movie and book recommender."""

import html
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="PlotFinder | Story recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the recommendation engine…")
def load_engine():
    # The recommender uses project-relative paths, so run from the project root.
    import os

    os.chdir(ROOT)
    from src import recommender

    return recommender


def genres_for(dataset: pd.DataFrame) -> list[str]:
    """Build the UI filter options from the existing dataset metadata."""
    from src.recommender import parse_genres

    values: set[str] = set()
    if "genres" in dataset.columns:
        for value in dataset["genres"].dropna():
            values.update(parse_genres(value))
    return sorted(values)


def display_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(part) for part in value)
    return str(value).strip()


def render_result(row: pd.Series, position: int, kind: str) -> None:
    title = html.escape(display_value(row.get("title")) or "Untitled")
    description = html.escape(display_value(row.get("description")) or "No plot description available.")
    genre_text = display_value(row.get("genres"))
    if genre_text.startswith("["):
        # CSV list columns are often represented as strings by pandas.
        try:
            import ast

            parsed = ast.literal_eval(genre_text)
            if isinstance(parsed, list):
                genre_text = ", ".join(map(str, parsed))
        except (ValueError, SyntaxError):
            pass
    genre_text = html.escape(genre_text)
    year_key = "release_year" if kind == "movie" else "publication_year"
    year = display_value(row.get(year_key))
    if year.endswith(".0"):
        year = year[:-2]
    rating = display_value(row.get("rating"))
    author = display_value(row.get("author"))
    score = row.get("similarity_score")

    facts = []
    if year:
        facts.append(f"{html.escape(year)}")
    if rating:
        facts.append(f"★ {html.escape(rating)}")
    if kind == "book" and author:
        facts.append(f"by {html.escape(author)}")
    if genre_text:
        facts.append(genre_text)
    facts_html = "<span> · </span>".join(facts)
    score_html = f'<span class="score">{float(score):.0%} match</span>' if pd.notna(score) else ""

    st.markdown(
        f"""
        <article class="result-card">
          <div class="result-top"><span class="result-number">{position:02d}</span>{score_html}</div>
          <h3>{title}</h3>
          <div class="facts">{facts_html}</div>
          <p>{description}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink:#17231e; --muted:#68766f; --paper:#f7f7f2; --green:#176b50; --line:#e4e8e1; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#eef2eb; border-right:1px solid var(--line); }
    .hero { padding:2.2rem 0 1.5rem; }
    .eyebrow { color:var(--green); font-size:.76rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .hero h1 { font-family:'Playfair Display',serif; font-size:clamp(2.5rem,5vw,4.2rem); letter-spacing:-.04em; line-height:1.04; margin:.45rem 0 .8rem; }
    .hero p { max-width:680px; color:var(--muted); font-size:1.08rem; line-height:1.7; }
    .result-card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.3rem 1.45rem; margin:0 0 1rem; box-shadow:0 5px 18px rgba(20,45,32,.035); }
    .result-top { display:flex; align-items:center; justify-content:space-between; }
    .result-number { color:#9aa69d; font-size:.78rem; font-weight:700; letter-spacing:.1em; }
    .score { color:var(--green); background:#e8f4ed; border-radius:99px; padding:.35rem .7rem; font-size:.76rem; font-weight:700; }
    .result-card h3 { font-family:'Playfair Display',serif; font-size:1.45rem; margin:.65rem 0 .25rem; color:var(--ink); }
    .facts { color:var(--green); font-size:.82rem; font-weight:600; }
    .result-card p { color:var(--muted); line-height:1.7; margin:.8rem 0 .1rem; }
    div.stButton > button[kind="primary"] { background:var(--green); border-color:var(--green); border-radius:10px; font-weight:700; }
    div.stButton > button[kind="primary"]:hover { background:#10563f; border-color:#10563f; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### 🎞️ PlotFinder")
    st.caption("A good story starts with a feeling.")
    st.markdown("---")
    content_type = st.radio("I’m looking for", ["Movies", "Books"], horizontal=True)
    result_count = st.slider("Recommendations", min_value=3, max_value=10, value=5)
    selected_genre = st.selectbox("Genre", ["Any genre"] + genres_for(
        load_engine().movies if content_type == "Movies" else load_engine().books
    ))
    st.markdown("---")
    st.caption("Describe a plot, mood, setting, or the kind of story you want to experience.")


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Find your next story</div>
      <h1>Tell us the plot.<br>We’ll find the feeling.</h1>
      <p>Describe the story you’re in the mood for, and discover movies and books with a similar plot.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.form("recommendation_form"):
    query = st.text_area(
        "What kind of story are you looking for?",
        placeholder="A lonely astronaut discovers a signal from Earth, but the message is decades old…",
        height=115,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Find my story  →", type="primary", use_container_width=True)


if submitted:
    if not query.strip():
        st.warning("Add a plot, mood, or story idea to get started.")
    else:
        engine = load_engine()
        is_movie = content_type == "Movies"
        recommend = engine.recommend_movies if is_movie else engine.recommend_books
        label = "movies" if is_movie else "books"
        with st.spinner(f"Searching for {label} that fit your description…"):
            try:
                results = recommend(
                    query,
                    genre=None if selected_genre == "Any genre" else selected_genre,
                    top_k=result_count,
                )
            except Exception as exc:
                st.error(f"The recommendation engine could not complete the search: {exc}")
                st.stop()

        st.markdown("### Your story matches")
        if results.empty:
            st.info("No matches came back for that search. Try a broader description or another genre.")
        else:
            st.caption(f"{len(results)} {label} inspired by your description")
            for position, (_, result) in enumerate(results.iterrows(), start=1):
                render_result(result, position, "movie" if is_movie else "book")
else:
    st.markdown("#### Need a prompt? Try one of these")
    suggestions = [
        "A clever heist with an unlikely team and a twist ending",
        "A quiet coming of age story set in a seaside town",
        "A dark fantasy about a kingdom where magic has a price",
    ]
    cols = st.columns(3)
    for col, suggestion in zip(cols, suggestions):
        with col:
            st.markdown(f"<div class='result-card'><p style='margin:0'>{html.escape(suggestion)}</p></div>", unsafe_allow_html=True)
