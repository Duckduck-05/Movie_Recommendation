import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collaborative_filtering import CF

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# List of genres
genres = [
    'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')



def recommend_movies(user_selected_genres, items, genres, top_n=10):
    """
    Recommend the top N movies based on the user's selected genres using content-based filtering.

    Args:
        user_selected_genres (list): List of genres selected by the user.
        items (pd.DataFrame): DataFrame containing movie data with genre columns.
        genres (list): List of all genres (columns of the `items` DataFrame).
        top_n (int): The number of top recommendations to return.

    Returns:
        pd.DataFrame: DataFrame of the top N recommended movies.
    """
    # Step 1: Create a binary vector for the user's genre preferences
    user_genre_vector = [1 if genre in user_selected_genres else 0 for genre in genres]
    user_genre_vector = np.array(user_genre_vector) / np.linalg.norm(user_genre_vector)  # Normalize vector

    # Step 2: Extract movie genre matrix (only the genre columns)
    X_train_counts = items[genres].values

    # Step 3: Calculate cosine similarity between the user's preferences and the movie genres
    similarity_scores = cosine_similarity([user_genre_vector], X_train_counts)

    # Step 4: Get the top N movie indices based on similarity scores
    recommended_movie_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

    # Step 5: Return the top N recommended movies
    recommended_movies = items.iloc[recommended_movie_indices]
    return recommended_movies


# Streamlit page setup
st.set_page_config(page_title="Film Genre Preference", page_icon=":movie_camera:", layout="centered")

# Display title and introduction
st.title("Welcome to Film Genre Preference")
st.write("""
    Select your preferred film genres from the list below. 
    Your choices will help us recommend the best films for you!
""")

# Checkbox for each genre
selected_genres = []
for genre in genres:
    if st.checkbox(genre):
        selected_genres.append(genre)


# Show the selected genres
if selected_genres:
    st.write("You selected the following genres:")
    st.write(", ".join(selected_genres))
else:
    st.write("Please select at least one genre to get film recommendations.")

# Model selection dropdown
model_type = st.radio(
    "Select Recommendation Model:",
    ('Content-Based', 'Collaborative Filtering')
)

# Button to submit and collect preferences
if st.button('Submit'):
    if selected_genres:
        st.write(f"Thank you for your preferences! You selected the {model_type} model.")
        st.write("Generating recommendations...")

        # Based on user selection, trigger the appropriate model
        if model_type == 'Content-Based':
            # Here, call your content-based recommendation system
            st.write("Using Content-Based Model to recommend films based on your selected genres...")

            # Get recommendations from content-based model
            recommended_movies = recommend_movies(selected_genres, items, genres)

            # Display recommended movies (Content-Based)
            st.write("Recommended Movies (Content-Based):")
            for idx, movie in recommended_movies.iterrows():
                st.write(f"**{movie['movie title']}**")
                st.write(f"Genres: {', '.join([genre for genre in genres if movie[genre] == 1])}")
                st.write(f"Links: {movie['IMDb URL']}")
                st.write("---")

        elif model_type == 'Collaborative Filtering':
            # Here, call your collaborative filtering recommendation system
            st.write("Using Collaborative Filtering Model to recommend films based on user preferences...")
            # You can call your collaborative filtering recommendation logic here, e.g.,
            rate_train = ratings_base.values
            rate_train[:, :2] -= 1  # Adjust for 0-indexing

            # Initialize CF model (user-user CF, k=30)
            cf_model = CF(rate_train, k=30, uuCF=1)  # Change `uuCF=0` for item-item CF
            cf_model.fit()  # Fit the model
            # Simulated recommendations for demonstration purposes
            recommended_movies = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
            st.write("Recommended Movies (Collaborative Filtering):")
            for movie in recommended_movies:
                st.write(movie)

    else:
        st.warning("You need to select at least one genre to submit your preferences.")
