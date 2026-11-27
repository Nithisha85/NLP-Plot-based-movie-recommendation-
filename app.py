"""Streamlit interface for the plot based movie and book recommender."""

import ast
import html
import os
from pathlib import Path

# Cap numerical-library thread pools before pandas imports NumPy. On Windows,
# OpenBLAS may otherwise reserve a large worker buffer for every reported core.
for thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_var] = "1"

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PlotFinder — stories for your mood",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Waking up the story finder…")
def load_engine():
    # The recommender uses project-relative paths, so run from the project root.
    os.chdir(ROOT)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from src import recommender
    recommender.faiss.omp_set_num_threads(1)

    return recommender


def genres_for(dataset: pd.DataFrame) -> list[str]:
    from src.recommender import parse_genres

    values: set[str] = set()
    if "genres" in dataset.columns:
        for value in dataset["genres"].dropna():
            values.update(parse_genres(value))
    return sorted(values, key=str.casefold)


def display_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)):
        return ", ".join(str(part) for part in value)
    return str(value).strip()


def genres_text(value) -> str:
    text = display_value(value)
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return ", ".join(map(str, parsed))
        except (ValueError, SyntaxError):
            pass
    return text


def render_result(row: pd.Series, position: int, kind: str) -> None:
    title = html.escape(display_value(row.get("title")) or "Untitled")
    description = html.escape(
        display_value(row.get("description")) or "No plot description available."
    )
    genre_text = html.escape(genres_text(row.get("genres")))
    year_key = "release_year" if kind == "movie" else "publication_year"
    year = display_value(row.get(year_key))
    if year.endswith(".0"):
        year = year[:-2]
    rating = display_value(row.get("rating"))
    author = display_value(row.get("author"))
    score = row.get("similarity_score")

    facts = []
    if year:
        facts.append(html.escape(year))
    if rating:
        facts.append(f"★ {html.escape(rating)}")
    if kind == "book" and author:
        facts.append(f"by {html.escape(author)}")
    facts_html = "<span class='fact-separator'>·</span>".join(facts)
    tags = "".join(
        f"<span class='genre-tag'>{html.escape(part.strip())}</span>"
        for part in genres_text(row.get("genres")).split(",")
        if part.strip()
    )
    score_html = (
        f"<span class='match-pill'>{float(score):.0%} match</span>"
        if pd.notna(score)
        else ""
    )
    st.markdown(
        f"""
        <article class="result-card">
          <div class="card-index">{position:02d}</div>
          <div class="card-content">
            <div class="card-overline">{kind.upper()} PICK {score_html}</div>
            <h3>{title}</h3>
            <div class="facts">{facts_html}</div>
            <p>{description}</p>
            <div class="genre-row">{tags}</div>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    :root { --ink:#20231f; --muted:#777970; --paper:#f5f3ed; --green:#315b48; --line:#e4e0d6; --orange:#d27a4a; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; transition:background 700ms ease; }
    .stApp:has(.movie-scene) { --ink:#f7eee3; --muted:#b4aaa0; --paper:#11131b; --green:#f0a15c; --line:rgba(255,235,210,.14); --orange:#ff684d; background:#11131b; }
    .stApp:has(.book-scene) { --ink:#f4ecd9; --muted:#b9ad95; --paper:#181611; --green:#a9bd8c; --line:rgba(245,225,185,.14); --orange:#d8a35d; background:#181611; }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1160px; padding-top:1.15rem; padding-bottom:5rem; position:relative; z-index:1; }
    .ambient-scene { position:fixed; inset:0; z-index:-1; overflow:hidden; pointer-events:none; isolation:isolate; background:#10131c; }
    .ambient-scene:before { content:""; position:absolute; inset:-15%; z-index:1; background:radial-gradient(ellipse at 50% 35%,rgba(91,75,106,.25),transparent 43%),radial-gradient(ellipse at 10% 80%,rgba(182,89,55,.1),transparent 35%),radial-gradient(ellipse at 90% 15%,rgba(51,110,143,.14),transparent 32%); animation:sceneGlow 12s ease-in-out infinite alternate; }
    .ambient-scene:after { content:""; position:absolute; inset:0; z-index:2; background:linear-gradient(90deg,rgba(11,13,19,.48),rgba(11,13,19,.25) 48%,rgba(11,13,19,.48)),linear-gradient(0deg,rgba(10,11,15,.72),transparent 42%,rgba(10,11,15,.34)); }
    .book-scene { background:radial-gradient(ellipse at 50% 60%,#55503a 0,#28251e 40%,#171611 78%); }
    .book-scene:before { background:radial-gradient(ellipse at 50% 42%,rgba(226,176,104,.23),transparent 42%),radial-gradient(ellipse at 20% 75%,rgba(79,108,76,.19),transparent 38%),radial-gradient(ellipse at 90% 15%,rgba(168,106,55,.14),transparent 32%); }
    .book-scene:after { background:linear-gradient(90deg,rgba(15,14,12,.25),rgba(15,14,12,.03) 48%,rgba(15,14,12,.3)),linear-gradient(0deg,rgba(14,13,11,.2),transparent 48%,rgba(14,13,11,.26)); }
    .cinema-video { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; object-position:center; filter:saturate(1.1) contrast(1.04); }
    .scene-grain { position:absolute; inset:0; opacity:.08; background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.4'/%3E%3C/svg%3E"); }
    @keyframes sceneGlow { from{transform:scale(1);opacity:.75} to{transform:scale(1.12);opacity:1} }
    .masthead { display:flex; align-items:center; justify-content:space-between; border:1px solid var(--line); border-radius:18px; padding:.85rem 1.1rem; margin-bottom:3.4rem; background:rgba(12,14,20,.38); backdrop-filter:blur(14px); box-shadow:0 12px 36px rgba(0,0,0,.14); }
    .brand { font-size:.82rem; font-weight:700; letter-spacing:.13em; color:var(--ink); }
    .brand-mark { color:var(--orange); margin-right:.55rem; }
    .mast-note { display:flex; align-items:center; gap:.6rem; color:var(--muted); font-size:.76rem; letter-spacing:.04em; }
    .live-dot { display:inline-block; width:7px; height:7px; flex:none; border-radius:50%; background:var(--green); box-shadow:0 0 13px var(--green); }
    .hero { max-width:900px; margin:0 auto 1.8rem; text-align:center; }
    .eyebrow { display:inline-flex; align-items:center; gap:.55rem; color:var(--ink); border:1px solid var(--line); background:rgba(16,17,22,.36); backdrop-filter:blur(10px); border-radius:999px; padding:.52rem .9rem; font-size:.67rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
    .hero h1 { font-family:'DM Serif Display',serif; font-weight:400; font-size:clamp(3.4rem,8vw,6.6rem); letter-spacing:-.045em; line-height:.93; margin:1.25rem 0 1.15rem; color:var(--ink); text-shadow:0 4px 36px rgba(0,0,0,.34); }
    .hero h1 em { color:var(--green); font-weight:400; text-shadow:0 0 40px color-mix(in srgb,var(--green),transparent 70%); }
    .hero p { max-width:600px; margin:0 auto; color:#ded8d1; font-size:1.02rem; line-height:1.75; text-shadow:0 2px 15px rgba(0,0,0,.7); }
    .input-heading { color:#f1e9df; font-size:.95rem; font-weight:600; margin:0 0 .55rem .15rem; }
    [data-testid="stForm"] { max-width:900px; margin:0 auto; background:linear-gradient(135deg,rgba(27,29,36,.82),rgba(15,17,22,.78)); backdrop-filter:blur(22px); border:1px solid rgba(255,255,255,.2); border-radius:26px; padding:1.35rem 1.5rem 1.2rem; box-shadow:0 24px 80px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.08); }
    [data-testid="stTextArea"] textarea { background:rgba(8,10,15,.56); color:var(--ink); border:1px solid rgba(255,255,255,.14); border-radius:16px; font-size:1rem; line-height:1.65; padding:1rem 1.1rem; transition:border-color .2s,box-shadow .2s; }
    [data-testid="stTextArea"] textarea:focus { border-color:var(--green); box-shadow:0 0 0 1px var(--green); }
    [data-testid="stFormSubmitButton"] button { min-height:3.25rem; border-radius:14px; background:linear-gradient(105deg,var(--green),color-mix(in srgb,var(--green),#fff 15%)); border-color:transparent; font-size:.95rem; font-weight:700; letter-spacing:.01em; box-shadow:0 8px 24px color-mix(in srgb,var(--green),transparent 72%); transition:transform .2s,box-shadow .2s; }
    [data-testid="stFormSubmitButton"] button:hover { filter:brightness(1.08); transform:translateY(-2px); box-shadow:0 12px 32px color-mix(in srgb,var(--green),transparent 62%); }
    div[data-testid="stRadio"] { max-width:900px; margin:0 auto 1.05rem; }
    div[data-testid="stRadio"] > label { font-size:.67rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#eee4da; }
    div[data-testid="stRadio"] [role="radiogroup"] { justify-content:center; gap:.6rem; }
    div[data-testid="stRadio"] [role="radiogroup"] label { background:rgba(16,17,22,.55); border:1px solid rgba(255,255,255,.2); border-radius:999px; padding:.48rem 1.15rem; transition:all .2s; }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background:color-mix(in srgb,var(--green),#17181e 77%); border-color:var(--green); box-shadow:0 0 22px color-mix(in srgb,var(--green),transparent 80%); }
    [data-testid="stSelectbox"] label, [data-testid="stSlider"] label, [data-testid="stSelectSlider"] label { font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#c8c0b6; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:rgba(10,12,16,.6); border-color:rgba(255,255,255,.16); border-radius:12px; }
    .try-label { max-width:900px; margin:1.45rem auto .65rem; color:#f0e8de; font-size:.68rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; text-align:center; }
    div[data-testid="stButton"] button { min-height:5.1rem; height:100%; white-space:normal; text-align:left; line-height:1.5; background:rgba(17,19,25,.49); color:#e1d9cf; border:1px solid rgba(255,255,255,.16); border-radius:15px; padding:.85rem 1rem; backdrop-filter:blur(13px); transition:transform .2s,border-color .2s,background .2s; }
    div[data-testid="stButton"] button:hover { transform:translateY(-3px); border-color:var(--green); background:rgba(28,30,36,.78); color:#fff; }
    .section-heading { display:flex; align-items:end; justify-content:space-between; margin:3.5rem 0 1rem; border-bottom:1px solid var(--line); padding-bottom:1rem; }
    .section-heading h2 { font-family:'DM Serif Display',serif; font-weight:400; font-size:2rem; margin:0; }
    .section-heading span { color:var(--muted); font-size:.8rem; }
    .result-card { position:relative; display:flex; gap:1.2rem; background:linear-gradient(135deg,rgba(27,29,36,.87),rgba(15,17,22,.84)); border:1px solid var(--line); border-radius:20px; padding:1.4rem 1.55rem; margin:.9rem 0; box-shadow:0 16px 45px rgba(0,0,0,.22); }
    .card-index { color:#c4c0b6; font-family:'DM Serif Display',serif; font-size:1.7rem; line-height:1; padding-top:.1rem; min-width:2.2rem; }
    .card-content { flex:1; min-width:0; }
    .card-overline { color:var(--green); font-size:.67rem; font-weight:700; letter-spacing:.13em; }
    .match-pill { float:right; background:#edf3ed; color:var(--green); border-radius:99px; letter-spacing:0; padding:.32rem .65rem; font-size:.7rem; }
    .result-card h3 { font-family:'DM Serif Display',serif; font-size:1.65rem; font-weight:400; margin:.35rem 0 .1rem; color:var(--ink); }
    .facts { color:var(--muted); font-size:.8rem; }
    .fact-separator { color:var(--orange); padding:0 .4rem; }
    .result-card p { color:#d0cac2; line-height:1.75; margin:.8rem 0 .85rem; font-size:.91rem; }
    .genre-row { display:flex; flex-wrap:wrap; gap:.4rem; }
    .genre-tag { background:rgba(255,255,255,.08); color:#d0c7bb; border:1px solid rgba(255,255,255,.08); border-radius:99px; padding:.3rem .65rem; font-size:.68rem; }
    [data-testid="stAlert"] { border-radius:12px; }
    .result-card { color:var(--ink); backdrop-filter:blur(15px); border-color:var(--line); }
    .result-card { animation:cardEnter .55s cubic-bezier(.2,.7,.2,1) both; transition:transform .2s,border-color .2s,box-shadow .2s; }
    .result-card:hover { transform:translateY(-2px); border-color:color-mix(in srgb,var(--green),transparent 48%); box-shadow:0 20px 54px rgba(0,0,0,.3); }
    .result-card h3 { color:var(--ink); }
    .match-pill { background:color-mix(in srgb,var(--green),transparent 84%); color:var(--green); }
    .card-index { color:rgba(255,255,255,.38); }
    .section-heading span { color:var(--muted); }
    @keyframes cardEnter { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
    @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; } }
    @media(max-width:640px) { .block-container{padding:1rem .8rem 3rem}.masthead{margin-bottom:2.1rem;padding:.75rem .85rem}.mast-note{font-size:.65rem}.hero{margin-bottom:1.5rem}.hero h1{font-size:clamp(3.2rem,14vw,4.2rem)}.hero p{font-size:.91rem;padding:0 .6rem}[data-testid="stForm"]{padding:1rem;border-radius:20px}.result-card{padding:1.1rem;gap:.6rem}.card-index{min-width:1.7rem}div[data-testid="stButton"] button{min-height:4.2rem;font-size:.76rem} }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = load_engine()
content_type = st.session_state.get("content_type") or "Movies"
movie_mode = content_type == "Movies"
if movie_mode:
    scene_class = "movie-scene"
    tagline = "A whole cinema of stories, just one search away."
    hero_copy = "Step into the scene. Describe the kind of movie world you want to disappear into."
    scene_asset = "movie-world-full.mp4"
else:
    scene_class = "book-scene"
    tagline = "Open a book. Step into another world."
    hero_copy = "Describe the world you want to disappear into, and find the story waiting there."
    scene_asset = "book-world.mp4"

scene_content = f"""
<video class="cinema-video" autoplay muted loop playsinline preload="auto" aria-hidden="true">
  <source src="/app/static/{scene_asset}" type="video/mp4">
</video>
"""
st.markdown(
    f'<div class="ambient-scene {scene_class}" aria-hidden="true">{scene_content}<div class="scene-grain"></div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="masthead"><div class="brand"><span class="brand-mark">✳</span>PLOTFINDER</div>
    <div class="mast-note"><span class="live-dot"></span>{html.escape(tagline)}</div></div>
    <section class="hero"><div class="eyebrow"><span class="live-dot"></span>{'THE MOVIE EXPERIENCE' if movie_mode else 'THE READING ROOM'}</div>
    <h1>{'A new world<br>starts with <em>a scene.</em>' if movie_mode else 'Turn the page.<br>Find <em>your world.</em>'}</h1>
    <p>{html.escape(hero_copy)}</p></section>
    """,
    unsafe_allow_html=True,
)

content_type = st.radio(
    "CHOOSE YOUR STORY FORMAT",
    ["Movies", "Books"],
    format_func=lambda value: "🎞️  Movies" if value == "Movies" else "📖  Books",
    horizontal=True,
    key="content_type",
)

with st.form("recommendation_form"):
    st.markdown(
        f"<div class='input-heading'>{'Describe the scene you want to watch' if movie_mode else 'Describe the world you want to read about'}</div>",
        unsafe_allow_html=True,
    )
    query = st.text_area(
        "Describe your story",
        placeholder=(
            "A lone pilot crosses a neon city as the sky begins to fold…"
            if movie_mode
            else "A forgotten book opens to a kingdom that should not exist…"
        ),
        height=110,
        label_visibility="collapsed",
        key="story_query",
    )
    filter_col, count_col = st.columns([2, 1])
    dataset = engine.movies if content_type == "Movies" else engine.books
    with filter_col:
        selected_genre = st.selectbox("Narrow by genre", ["Any genre"] + genres_for(dataset))
    with count_col:
        result_count = st.select_slider("How many picks?", options=[3, 5, 7, 10], value=5)
    submit_label = "Find my movie  →" if movie_mode else "Find my next book  →"
    submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)

prompt_heading = "Need a scene? Try one of these" if movie_mode else "Plant a story seed"
st.markdown(f'<div class="try-label">{prompt_heading}</div>', unsafe_allow_html=True)
suggestions = (
    [
        "A neon-lit city, one impossible chase, and a secret hidden in the rain",
        "An intimate space odyssey where two strangers hear the same signal",
        "A heist crew has one night to steal something that does not exist",
    ]
    if movie_mode
    else [
        "An enchanted atlas leads a lonely cartographer to a vanished kingdom",
        "A little bookshop at the edge of the sea keeps changing its stories",
        "A child finds a door in the library that opens only at midnight",
    ]
)
def apply_suggestion(value: str) -> None:
    st.session_state["story_query"] = value


cols = st.columns(3)
for index, (col, suggestion) in enumerate(zip(cols, suggestions)):
    with col:
        st.button(
            suggestion,
            key=f"suggestion_{content_type}_{index}",
            on_click=apply_suggestion,
            args=(suggestion,),
            use_container_width=True,
        )

if submitted:
    if not query.strip():
        st.warning("Add a plot, mood, or story idea to get started.")
    else:
        is_movie = content_type == "Movies"
        recommend = engine.recommend_movies if is_movie else engine.recommend_books
        label = "movies" if is_movie else "books"
        with st.spinner(f"Finding {label} that fit your description…"):
            try:
                results = recommend(
                    query,
                    genre=None if selected_genre == "Any genre" else selected_genre,
                    top_k=result_count,
                )
            except Exception as exc:
                st.error(f"The recommendation engine could not complete the search: {exc}")
                st.stop()

        st.markdown(
            f'<div class="section-heading"><h2>Your story matches</h2><span>{len(results)} {label} for this mood</span></div>',
            unsafe_allow_html=True,
        )
        if results.empty:
            st.info("No matches came back for that search. Try a broader description or another genre.")
        else:
            for position, (_, result) in enumerate(results.iterrows(), start=1):
                render_result(result, position, "movie" if is_movie else "book")
