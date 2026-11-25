
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# ==========================================
# 1. LOAD CLEANED DATASETS
# ==========================================

print("Loading cleaned datasets...")

movies = pd.read_csv("datasets/cleaned_movies.csv")
books = pd.read_csv("datasets/cleaned_books.csv")

# ==========================================
# 2. LOAD SBERT MODEL
# ==========================================

print("\nLoading SBERT model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("SBERT model loaded successfully!")

# ==========================================
# 3. GENERATE MOVIE EMBEDDINGS
# ==========================================

print("\nGenerating movie embeddings...")

movie_embeddings = model.encode(
    movies["description"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Movie embeddings generated!")
print("Movie embedding shape:", movie_embeddings.shape)

# ==========================================
# 4. GENERATE BOOK EMBEDDINGS
# ==========================================

print("\nGenerating book embeddings...")

book_embeddings = model.encode(
    books["description"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Book embeddings generated!")
print("Book embedding shape:", book_embeddings.shape)

# ==========================================
# 5. SAVE EMBEDDINGS
# ==========================================

os.makedirs("models", exist_ok=True)

np.save("models/movie_embeddings.npy", movie_embeddings)
np.save("models/book_embeddings.npy", book_embeddings)

print("\nEmbeddings saved successfully!")

print("Movie embeddings:", movie_embeddings.shape)
print("Book embeddings:", book_embeddings.shape)
