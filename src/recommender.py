
import pandas as pd
import numpy as np
import faiss
import ast
from sentence_transformers import SentenceTransformer

print("Loading recommendation engine...")

# Load cleaned datasets
movies = pd.read_csv("datasets/cleaned_movies.csv")
books = pd.read_csv("datasets/cleaned_books.csv")

# Load saved FAISS indexes
movie_index = faiss.read_index("models/movie_index.faiss")
book_index = faiss.read_index("models/book_index.faiss")

# Load the same SBERT model used for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Recommendation engine loaded successfully!")


# Convert stored genre strings into Python lists
def parse_genres(genre_value):
    try:
        genres = ast.literal_eval(genre_value)

        if isinstance(genres, list):
            return [str(g).strip().lower() for g in genres]

        return []

    except (ValueError, SyntaxError, TypeError):
        return []


# Generate embedding for the user's search query
def generate_query_embedding(query):
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.ascontiguousarray(embedding, dtype=np.float32)


# Generic recommendation function
def get_recommendations(query, dataset, index, genre=None, top_k=5):

    if not query or not query.strip():
        raise ValueError("Please enter a search description.")

    # Convert query into an embedding
    query_embedding = generate_query_embedding(query)

    # Search all vectors so genre filtering doesn't miss matches
    scores, indices = index.search(
        query_embedding,
        index.ntotal
    )

    recommendations = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        item = dataset.iloc[idx]

        # Apply optional genre filter
        if genre:
            item_genres = parse_genres(item["genres"])

            if genre.strip().lower() not in item_genres:
                continue

        result = item.to_dict()
        result["similarity_score"] = float(score)

        recommendations.append(result)

        if len(recommendations) == top_k:
            break

    return pd.DataFrame(recommendations)


# Recommend movies
def recommend_movies(query, genre=None, top_k=5):

    return get_recommendations(
        query=query,
        dataset=movies,
        index=movie_index,
        genre=genre,
        top_k=top_k
    )


# Recommend books
def recommend_books(query, genre=None, top_k=5):

    return get_recommendations(
        query=query,
        dataset=books,
        index=book_index,
        genre=genre,
        top_k=top_k
    )


if __name__ == "__main__":

    test_query = "A mind-bending science fiction story involving time travel and alternate realities"

    print("\nTesting movie recommendations...")
    movie_results = recommend_movies(test_query, top_k=5)

    print(movie_results[
        ["title", "genres", "similarity_score"]
    ].to_string(index=False))

    print("\nTesting book recommendations...")
    book_results = recommend_books(test_query, top_k=5)

    print(book_results[
        ["title", "genres", "similarity_score"]
    ].to_string(index=False))
