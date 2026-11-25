
import numpy as np
import pandas as pd
import faiss
import os

# ==========================================
# 1. LOAD EMBEDDINGS
# ==========================================

print("Loading embeddings...")

movie_embeddings = np.load("models/movie_embeddings.npy")
book_embeddings = np.load("models/book_embeddings.npy")

# ==========================================
# 2. LOAD METADATA
# ==========================================

movies = pd.read_csv("datasets/cleaned_movies.csv")
books = pd.read_csv("datasets/cleaned_books.csv")

# ==========================================
# 3. CREATE MOVIE FAISS INDEX
# ==========================================

print("\nCreating movie FAISS index...")

dimension = movie_embeddings.shape[1]

movie_index = faiss.IndexFlatIP(dimension)

movie_index.add(
    np.ascontiguousarray(movie_embeddings, dtype=np.float32)
)

print("Movie index created!")
print("Total movie vectors:", movie_index.ntotal)

# ==========================================
# 4. CREATE BOOK FAISS INDEX
# ==========================================

print("\nCreating book FAISS index...")

book_index = faiss.IndexFlatIP(dimension)

book_index.add(
    np.ascontiguousarray(book_embeddings, dtype=np.float32)
)

print("Book index created!")
print("Total book vectors:", book_index.ntotal)

# ==========================================
# 5. SAVE FAISS INDEXES
# ==========================================

os.makedirs("models", exist_ok=True)

faiss.write_index(
    movie_index,
    "models/movie_index.faiss"
)

faiss.write_index(
    book_index,
    "models/book_index.faiss"
)

print("\nFAISS indexes saved successfully!")

# ==========================================
# 6. VERIFY INDEXES
# ==========================================

print("\nVerification:")

print("Movie index dimension:", movie_index.d)
print("Book index dimension:", book_index.d)

print("Movie vectors:", movie_index.ntotal)
print("Book vectors:", book_index.ntotal)
