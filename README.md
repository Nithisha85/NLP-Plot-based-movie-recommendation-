# Plot-Based Movie and Book Recommender

Find a movie or book by describing a plot, setting, character, or mood. PlotFinder turns a natural-language search into recommendations from separate movie and book catalogs.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Key Features](#key-features)
5. [System Architecture](#system-architecture)
6. [Workflow](#workflow)
7. [Technology Stack](#technology-stack)
8. [Datasets](#datasets)
9. [Project Structure](#project-structure)
10. [How It Works](#how-it-works)
11. [Installation and Setup](#installation-and-setup)
12. [Running the Algorithm](#running-the-algorithm)
13. [Example Search](#example-search)
14. [Evaluation and Testing](#evaluation-and-testing)
15. [Limitations](#limitations)
16. [Future Improvements](#future-improvements)
17. [Acknowledgements](#acknowledgements)

## Project Overview

PlotFinder is a semantic recommendation app for movies and books. Instead of requiring an exact title or keyword, it lets a person describe the kind of story they want. The app embeds the description and compares it with plot summaries in the selected catalog.

The Streamlit interface includes separate Movies and Books experiences, genre filtering, adjustable result counts, similarity scores, and plot metadata. Each mode has its own looping video background.

## Problem Statement

Traditional catalog searches depend on matching words in a title or a small set of tags. They can miss useful results when someone describes a mood, situation, or story idea using different words from the catalog description.

This project uses sentence embeddings to find plot summaries with related meaning, even when the query and catalog use different wording.

## Objectives

- Recommend movies and books from a free-form plot or mood description.
- Represent queries and catalog descriptions in the same semantic vector space.
- Rank results by vector similarity and optionally filter them by genre.
- Show useful title, plot, and available metadata in an interactive interface.
- Keep movie and book catalogs and indexes separate.

## Key Features

- Natural-language search across movie and book plot summaries.
- Separate movie and book catalogs with theme-specific backgrounds.
- Optional genre filter and configurable result count.
- Similarity score, year, rating, genres, and book author where available.
- Clickable example prompts to start a search.
- Prebuilt FAISS indexes so catalog vectors do not need to be regenerated at app startup.

## System Architecture

```mermaid
flowchart LR
    U[User story description] --> A[Streamlit interface]
    A --> E[Sentence Transformers encoder]
    E --> Q[Normalized query vector]
    Q --> FM[Movie FAISS index]
    Q --> FB[Book FAISS index]
    FM --> RM[Ranked movie row IDs]
    FB --> RB[Ranked book row IDs]
    RM --> CM[Movie CSV metadata]
    RB --> CB[Book CSV metadata]
    CM --> O[Genre filter and top results]
    CB --> O
    O --> UI[Recommendation cards]
```

Movie and book searches use the same embedding model but load their own cleaned CSV files and FAISS indexes. The CSV row order must match the order used when the corresponding index was built.

## Workflow

1. Choose Movies or Books.
2. Enter a plot, setting, character, or mood, or choose an example prompt.
3. Optionally select a genre and result count.
4. Encode the search text with `all-MiniLM-L6-v2` and normalize its vector.
5. Search the selected FAISS index by inner product.
6. Apply the optional genre filter and return the highest-scoring matches.
7. Display each result with its available metadata and plot summary.

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| User interface | Streamlit | Search form, filters, result display |
| Language embeddings | Sentence Transformers, `all-MiniLM-L6-v2` | Encodes queries and plot descriptions |
| Vector search | FAISS `IndexFlatIP` | Exact inner-product ranking over normalized vectors |
| Data handling | pandas, NumPy | Loads catalogs and prepares vectors |
| Inference runtime | PyTorch | Runs the embedding model |
| Background media | Streamlit static-file serving | Serves the movie and book theme videos |

## Datasets

| File | Contents | Runtime use |
|---|---|---|
| `datasets/cleaned_movies.csv` | 1,000 movie records mapped from the supplied IMDb Top 1000 CSV | Movie titles, descriptions, genres, years, ratings, and vote counts |
| `datasets/cleaned_books.csv` | 9,943 cleaned book records | Book titles, descriptions, genres, years, ratings, and authors |
| `datasets/books_enriched.csv` | Enriched book source data | Source for the cleaned book catalog |
| `datasets/tmdb_5000_movies.csv` | TMDB movie data retained in the repository | Legacy source for the existing preprocessing script; not the current runtime catalog |

The original `imdb_top_1000.csv` import is represented by `cleaned_movies.csv`; the raw IMDb CSV is not included in this repository. The movie catalog has a matching 1,000 × 384 embedding matrix and FAISS index. The book catalog has a matching 9,943 × 384 matrix and index.

Both background videos are in `static/`: `movie-world-full.mp4` for Movies and `book-world.mp4` for Books. Static serving is enabled in `.streamlit/config.toml`.

## Project Structure

```text
app.py                         Streamlit interface
requirements.txt               Python dependencies
.streamlit/config.toml         Enables static-file serving
src/recommender.py             Loads catalogs, model, indexes, and runs searches
src/data_preprocessing.py      Builds cleaned catalogs from source CSV files
src/embedding_generator.py     Creates movie and book embedding matrices
src/vector_search.py           Builds and saves movie and book FAISS indexes
datasets/
  cleaned_movies.csv           Current IMDb Top 1000 movie catalog
  cleaned_books.csv            Cleaned book catalog
  books_enriched.csv           Book source data
  tmdb_5000_movies.csv          Legacy movie source data
models/
  movie_embeddings.npy         Movie vectors
  movie_index.faiss             Movie search index
  book_embeddings.npy          Book vectors
  book_index.faiss              Book search index
static/
  movie-world-full.mp4          Movies background video
  book-world.mp4                Books background video
notebooks/
  01_data_exploration.ipynb     Dataset exploration
```

## How It Works

The model maps the query and each plot description to 384-dimensional vectors. Both are normalized, so their inner product is equivalent to cosine similarity. FAISS returns candidate row positions in descending similarity order. The recommender uses those positions to retrieve metadata from the selected CSV, applies a genre filter if requested, and stops after collecting the requested number of results.

The displayed match score is the raw vector similarity formatted as a percentage for readability. It is not a calibrated probability that someone will like the title.

## Installation and Setup

Use Python 3.12 for local development and to match the deployment runtime. From the project root, create and activate a virtual environment, then install the dependencies.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Before running, limit numerical-library thread counts to avoid excessive worker allocation on machines reporting many CPU cores:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The first startup downloads or loads `all-MiniLM-L6-v2` from the Hugging Face Hub. Internet access is required if the model is not already cached.

### Streamlit Community Cloud

The GitHub repository contains the app entry point, root `requirements.txt`, configuration, data, indexes, and video assets. In Streamlit Community Cloud, create an app from the repository, choose branch `main`, set the entry point to `app.py`, and select Python 3.12 in Advanced settings.

## Running the Algorithm

Start the interactive app from the project root:

```bash
streamlit run app.py
```

For a command-line smoke run of both recommenders, use:

```bash
python src/recommender.py
```

The script runs a sample science-fiction query and prints movie and book titles, genres, and similarity scores.

To rebuild embeddings and indexes from the existing cleaned catalogs, run:

```bash
python src/embedding_generator.py
python src/vector_search.py
```

These scripts rebuild both movie and book artifacts. **Do not run `src/data_preprocessing.py` if you want to keep the current IMDb movie catalog:** that script currently reads `tmdb_5000_movies.csv` and overwrites `cleaned_movies.csv` with the TMDB-derived catalog.

## Example Search

**Search description:**

> A mind-bending science-fiction story involving time travel and alternate realities.

Choose Movies or Books, optionally choose the `science fiction` genre, and request five results. PlotFinder returns the closest matching summaries in that catalog, along with their available metadata and similarity scores.

## Evaluation and Testing

The repository does not include a labeled relevance set or report formal ranking metrics such as precision@k, recall@k, or NDCG. The current evaluation method is a manual smoke run using `python src/recommender.py` and example queries in the interface.

Useful manual checks include:

- Confirm the movie index has 1,000 vectors and the book index has 9,943.
- Try a query with and without a genre filter.
- Confirm the selected catalog returns no more than the requested number of results.
- Try an empty query and confirm the interface asks for a description.
- Review whether the top results match the meaning of the query, not only its wording.

## Limitations

- Recommendations depend on the coverage and quality of available plot descriptions and genres.
- The movie catalog contains 1,000 titles; the book catalog contains 9,943 titles.
- Similarity scores measure vector proximity and are not calibrated user-preference probabilities.
- There is no user profile, watch/read history, collaborative filtering, or feedback loop.
- Search requires the embedding model to be available locally or downloadable at startup.
- `src/data_preprocessing.py` still targets the legacy TMDB movie source and can overwrite the current IMDb-derived catalog if run.

## Future Improvements

- Add an evaluation dataset with human relevance judgments and report ranking metrics.
- Add more movie and book records while keeping CSV rows and vector indexes synchronized.
- Add poster or cover art, richer filters, and clickable detail links.
- Support user feedback and optional preference-based reranking.
- Update preprocessing so it can rebuild the current IMDb-derived movie catalog from a checked-in or user-provided raw file.
- Pin and regularly validate dependency versions for repeatable local and cloud deployments.

## Acknowledgements

- The supplied IMDb Top 1000 movie dataset and the included enriched book and TMDB datasets.
- Sentence Transformers and the `all-MiniLM-L6-v2` model for semantic embeddings.
- FAISS for vector similarity search and Streamlit for the interactive application.

Please follow the original source and license terms for any datasets or media you redistribute.
