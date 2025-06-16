# books/recommender.py

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- Set up paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'static', 'data')

# --- Load data files ---
books = pd.read_csv(
    os.path.join(DATA_PATH, 'Books.csv'),
    on_bad_lines='skip',
    encoding='latin-1',
    low_memory=False
)[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.columns = ['isbn', 'title', 'author', 'year', 'publisher']

ratings = pd.read_csv(
    os.path.join(DATA_PATH, 'Ratings.csv'),
    on_bad_lines='skip',
    encoding='latin-1',
    low_memory=False
)
ratings.rename(columns={'User-ID': 'user_id', 'ISBN': 'isbn', 'Book-Rating': 'books_rating'}, inplace=True)

# --- Preprocessing ---
active_users = ratings['user_id'].value_counts()[ratings['user_id'].value_counts() > 200].index
ratings = ratings[ratings['user_id'].isin(active_users)]
ratings = ratings.merge(books, on='isbn')

rating_counts = ratings.groupby('title')['books_rating'].count().reset_index()
rating_counts.rename(columns={'books_rating': 'number_of_ratings'}, inplace=True)

final_data = ratings.merge(rating_counts, on='title')
final_data = final_data[final_data['number_of_ratings'] >= 50]
final_data.drop_duplicates(['user_id', 'title'], inplace=True)

# --- Create pivot and train model ---
book_pivot = final_data.pivot_table(columns='user_id', index='title', values='books_rating').fillna(0)
book_sparse = csr_matrix(book_pivot)

model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# --- Datalist support ---
book_titles = list(book_pivot.index)

# --- Utility Functions ---
def get_title_by_index(index):
    if 0 <= index < len(book_titles):
        return book_titles[index]
    else:
        raise IndexError("Index out of range.")

def recommend_books(book_name, n_recommendations=5):
    if book_name not in book_pivot.index:
        raise ValueError("Book not found.")
    idx = book_pivot.index.get_loc(book_name)
    distances, suggestions = model.kneighbors(book_pivot.iloc[idx, :].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
    return [book_pivot.index[i] for i in suggestions[0][1:]]