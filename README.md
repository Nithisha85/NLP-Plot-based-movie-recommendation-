# Plot Based Movie and Book Recommender

Find movies and books by describing a plot, setting, or mood. The app embeds the description with Sentence Transformers and searches the existing FAISS indexes for similar titles.

## Features

- Search movie and book plots with a natural language description.
- Filter recommendations by genre and choose how many results to show.
- View plot summaries and available title metadata alongside similarity scores.

## Requirements

- Python 3.10 or newer
- The datasets and model artifacts included in this project
- Internet access on the first run if the `all-MiniLM-L6-v2` Sentence Transformers model is not already cached

## Setup

Open a terminal in the project directory, create and activate a virtual environment, and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run the app

Start Streamlit from the project root:

```powershell
streamlit run app.py
```

Streamlit will print a local URL and usually open it in your browser. Do not start the interface with `python app.py`; Streamlit apps need to be launched with `streamlit run`.

## Project layout

```text
app.py                         Streamlit user interface
src/recommender.py             Existing recommendation functions and model loading
src/data_preprocessing.py      Prepare cleaned movie and book datasets
src/embedding_generator.py     Generate text embeddings
src/vector_search.py            Build and save FAISS indexes
datasets/                      Source and cleaned CSV datasets
models/                         Embeddings and saved FAISS indexes
notebooks/                      Data exploration notebook
```

The app expects to run from the project root and uses these files:

- `datasets/cleaned_movies.csv`
- `datasets/cleaned_books.csv`
- `models/movie_index.faiss`
- `models/book_index.faiss`

The Sentence Transformers model is `all-MiniLM-L6-v2`. It is downloaded or loaded from the local Hugging Face cache when the recommendation engine starts.

## Rebuild the recommendation artifacts

The checked-in/packaged cleaned data and FAISS indexes are ready to use. To regenerate them from the source CSV files, run these scripts from the project root, in order:

```powershell
python src/data_preprocessing.py
python src/embedding_generator.py
python src/vector_search.py
```

Embedding generation downloads or loads `all-MiniLM-L6-v2` and can take some time. This rebuild workflow does not change the recommendation method used by the app.

## Notebook

To run the data exploration notebook, install the project requirements, then launch Jupyter:

```powershell
python -m pip install jupyter
jupyter notebook notebooks/01_data_exploration.ipynb
```
