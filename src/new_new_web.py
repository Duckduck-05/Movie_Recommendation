import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from lib.collaborative_filtering import CF
from lib.matrix_factorization import MatrixFactorization  # Import the MF.py module

# Define movie data columns
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# List of genres (columns for genre-based filtering)
genres = [
    'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# Load movie data
items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

# Map movie ids to movie titles
movies_set_raw = pd.read_csv('./data/ml-100k/u.item', encoding="latin-1", sep="|", usecols=[0, 1], names=["movie_id", "movie_name"])
movie_dict = pd.Series(movies_set_raw['movie_name'].values, index=movies_set_raw['movie_id']).to_dict()

# Load ratings data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('./data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

# Function to recommend movies for new users based on genre preferences
def recommend_movies_for_new_user(user_selected_genres, items, genres, top_n=10):
    user_genre_vector = [1 if genre in user_selected_genres else 0 for genre in genres]
    user_genre_vector = np.array(user_genre_vector) / np.linalg.norm(user_genre_vector)

    X_train_counts = items[genres].values
    similarity_scores = cosine_similarity([user_genre_vector], X_train_counts)
    recommended_movie_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

    recommended_movies = items.iloc[recommended_movie_indices]
    return recommended_movies

# Function to recommend movies for current users based on collaborative filtering
def recommend_movies_for_current_user(user_id, ratings_base, items, k=30, top_n=10):
    rate_train = ratings_base.values
    rate_train[:, :2] -= 1  # Adjust for 0-indexing

    cf_model = CF(rate_train, k=k, uuCF=1)
    cf_model.fit()
    recommended_movie_ids = cf_model.recommend(user_id)[:top_n]

    recommended_movies = items.iloc[recommended_movie_ids]
    return recommended_movies

# Function to get the movies watched by a current user
def get_watched_movies(user_id, ratings_base, items):
    user_ratings = ratings_base[ratings_base['user_id'] == user_id]
    watched_movie_ids = user_ratings['movie_id'].unique()
    watched_movies = items[items['movie id'].isin(watched_movie_ids)]
    return watched_movies

# Streamlit page setup
st.set_page_config(page_title="Film Genre Preference", page_icon=":movie_camera:", layout="centered")

# Display title and introduction
st.title("Welcome to Film Genre Preference")
st.write("""
    Please select whether you are a **new user** or a **current user**.
""")

# User selection: new_user or current_user
user_type = st.radio("Are you a new user or a current user?", ('New User', 'Current User'))

if user_type == 'New User':
    st.write("As a new user, please select your preferred film genres.")

    # Checkbox for each genre
    selected_genres = []
    for genre in genres:
        if st.checkbox(genre):
            selected_genres.append(genre)

    # Display selected genres
    if selected_genres:
        st.write(f"You selected the following genres: {', '.join(selected_genres)}")

        # Button to submit and get recommendations
        if st.button('Get Recommendations'):
            st.write("Generating movie recommendations based on your selected genres...")
            recommended_movies = recommend_movies_for_new_user(selected_genres, items, genres)

            # Display recommended movies
            st.write("Recommended Movies:")
            for idx, movie in recommended_movies.iterrows():
                st.write(f"**{movie['movie title']}**")
                st.write(f"Genres: {', '.join([genre for genre in genres if movie[genre] == 1])}")
                st.write(f"Link: {movie['IMDb URL']}")
                st.write("---")

    else:
        st.write("Please select at least one genre to get movie recommendations.")

elif user_type == 'Current User':
    st.write("As a current user, please select your user ID.")

    # Dropdown to select user ID (simulate user ID selection)
    user_id = st.number_input("Enter your user ID:", min_value=0, max_value=943, step=1)

    # Show the movies the current user has watched
    if st.button('Show Watched Movies'):
        st.write(f"Movies that user {user_id} has watched:")

        # Get and display the movies the current user has watched
        watched_movies = get_watched_movies(user_id, ratings_base, items)
        if not watched_movies.empty:
            for idx, movie in watched_movies.iterrows():
                st.write(f"**{movie['movie title']}**")
                st.write(f"Genres: {', '.join([genre for genre in genres if movie[genre] == 1])}")
                st.write(f"Link: {movie['IMDb URL']}")
                st.write("---")
        else:
            st.write("No movies found for this user.")

    # Button to submit and get recommendations
    if st.button('Get Recommendations'):
        st.write(f"Generating movie recommendations for user {user_id}...")
        recommended_movies = recommend_movies_for_current_user(user_id, ratings_base, items)

        # Display recommended movies
        st.write("Recommended Movies:")
        for idx, movie in recommended_movies.iterrows():
            st.write(f"**{movie['movie title']}**")
            st.write(f"Genres: {', '.join([genre for genre in genres if movie[genre] == 1])}")
            st.write(f"Link: {movie['IMDb URL']}")
            st.write("---")

    # Add Matrix Factorization button
    if st.button('Get Matrix Factorization Recommendations'):
        st.write(f"Generating movie recommendations using Matrix Factorization for user {user_id}...")

        # Use the MatrixFactorization class from MF.py
        mf_model = MatrixFactorization(ratings_base, num_latent_factors=20)  # Example: Using 20 latent factors
        mf_model.fit()  # Fit the model
        recommended_movies, _ = mf_model.recommend(user_id, top_n=10)  # Get top 10 recommended movies

        # Display recommended movies using Matrix Factorization
        st.write("Recommended Movies using Matrix Factorization:")
        for movie_id in recommended_movies:
            movie = items[items['movie id'] == movie_id].iloc[0]
            st.write(f"**{movie['movie title']}**")
            st.write(f"Genres: {', '.join([genre for genre in genres if movie[genre] == 1])}")
            st.write(f"Link: {movie['IMDb URL']}")
            st.write("---")
