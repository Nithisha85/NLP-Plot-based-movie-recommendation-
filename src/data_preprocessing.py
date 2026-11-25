import pandas as pd
import numpy as np
import ast
import json
import re


# ==========================================
# 1. LOAD DATASETS
# ==========================================

print("Loading datasets...")

movies = pd.read_csv("datasets/tmdb_5000_movies.csv")
books = pd.read_csv("datasets/books_enriched.csv")

print("Datasets loaded successfully!")


# ==========================================
# 2. GENRE EXTRACTION FUNCTION
# ==========================================

def extract_genres(value):

    if pd.isna(value):
        return []

    try:
        if isinstance(value, list):
            data = value
        else:
            data = ast.literal_eval(value)

        if not isinstance(data, list):
            return []

        genres = []

        for item in data:

            if isinstance(item, dict):
                genre = item.get("name", "")

            elif isinstance(item, str):
                genre = item

            else:
                continue

            if genre:
                genres.append(genre.strip().replace("-", " ").title())

        return list(dict.fromkeys(genres))

    except (ValueError, SyntaxError, TypeError):
        return []


# ==========================================
# 3. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================================
# 4. MOVIE PREPROCESSING
# ==========================================

print("\nPreprocessing movies...")

clean_movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres",
        "release_date",
        "vote_average",
        "popularity"
    ]
].copy()

# Rename columns
clean_movies.rename(
    columns={
        "id": "movie_id",
        "overview": "description",
        "release_date": "release_year",
        "vote_average": "rating"
    },
    inplace=True
)

# Remove duplicates
clean_movies.drop_duplicates(subset="movie_id", inplace=True)

# Remove missing titles and descriptions
clean_movies.dropna(
    subset=["title", "description"],
    inplace=True
)

# Remove empty descriptions
clean_movies = clean_movies[
    clean_movies["description"].astype(str).str.strip() != ""
]

# Clean descriptions
clean_movies["description"] = clean_movies["description"].apply(clean_text)

# Extract genre names
clean_movies["genres"] = clean_movies["genres"].apply(extract_genres)

# Extract release year
clean_movies["release_year"] = pd.to_datetime(
    clean_movies["release_year"],
    errors="coerce"
).dt.year

# Reset index
clean_movies.reset_index(drop=True, inplace=True)


# ==========================================
# 5. BOOK PREPROCESSING
# ==========================================

print("\nPreprocessing books...")

clean_books = books[
    [
        "book_id",
        "title",
        "authors",
        "description",
        "genres",
        "original_publication_year",
        "average_rating"
    ]
].copy()

# Rename columns
clean_books.rename(
    columns={
        "authors": "author",
        "original_publication_year": "publication_year",
        "average_rating": "rating"
    },
    inplace=True
)

# Remove duplicates
clean_books.drop_duplicates(subset="book_id", inplace=True)

# Remove missing titles and descriptions
clean_books.dropna(
    subset=["title", "description"],
    inplace=True
)

# Remove empty descriptions
clean_books = clean_books[
    clean_books["description"].astype(str).str.strip() != ""
]

# Clean descriptions
clean_books["description"] = clean_books["description"].apply(clean_text)

# Extract genre names
clean_books["genres"] = clean_books["genres"].apply(extract_genres)

# Convert publication year to numeric
clean_books["publication_year"] = pd.to_numeric(
    clean_books["publication_year"],
    errors="coerce"
)

# Reset index
clean_books.reset_index(drop=True, inplace=True)


# ==========================================
# 6. SAVE CLEANED DATASETS
# ==========================================

clean_movies.to_csv(
    "datasets/cleaned_movies.csv",
    index=False
)

clean_books.to_csv(
    "datasets/cleaned_books.csv",
    index=False
)

print("\nPreprocessing completed successfully!")

print("\nMovie dataset shape:", clean_movies.shape)
print("Book dataset shape:", clean_books.shape)

print("\nCleaned movie dataset:")
print(clean_movies.head())

print("\nCleaned book dataset:")
print(clean_books.head())

print("\nFiles saved successfully!")