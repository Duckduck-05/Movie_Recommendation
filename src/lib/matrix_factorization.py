import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

class MatrixFactorization:
    def __init__(self, ratings, num_latent_factors=20):
        self.ratings = ratings
        self.num_latent_factors = num_latent_factors
        self.R = self.create_rating_matrix()
        self.num_users, self.num_items = self.R.shape
        self.P = np.random.rand(self.num_users, self.num_latent_factors)
        self.Q = np.random.rand(self.num_items, self.num_latent_factors)

    def create_rating_matrix(self):
        return self.ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    def normalize(self):
        mask = np.equal(self.R, 0)
        masked_arr = np.ma.masked_array(self.R, mask)
        temp_mask = masked_arr.T
        rating_means = np.mean(temp_mask, axis=0)
        filled_matrix = temp_mask.filled(rating_means)
        filled_matrix = filled_matrix.T
        filled_matrix -= rating_means.data[:, np.newaxis]
        self.normalized = filled_matrix
        self.rating_means = rating_means
        return self.normalized

    def apply_svd_with_dim(self, latent_dim):
        user_f, features, f_movie = svds(self.normalized, k=latent_dim)
        feature_diag_matrix = np.diag(features) if features.ndim == 1 else np.diag(features[0])
        return user_f, feature_diag_matrix, f_movie

    def generate_predictions(self, user_f, feature_diag_matrix, f_movie):
        predicted_ratings = np.dot(np.dot(user_f, feature_diag_matrix), f_movie)
        preds_df = pd.DataFrame(predicted_ratings, columns=self.R.columns, index=self.R.index)
        return preds_df

    def fit(self):
        self.normalize()
        user_feature, feature_diag_matrix, movie_feature = self.apply_svd_with_dim(self.num_latent_factors)
        df_predictions = self.generate_predictions(user_feature, feature_diag_matrix, movie_feature)
        self.prediction_normal = df_predictions
        self.prediction = df_predictions + self.rating_means.data[:, np.newaxis]
        return user_feature, feature_diag_matrix, movie_feature, df_predictions

    def predict(self, user_id, item_id):
        return self.prediction.iloc[user_id-1][item_id]

    def recommend(self, user_id, top_n=10):
        user_id = user_id - 1
        prediction_arr = self.prediction.to_numpy()
        predicted_ratings_user = prediction_arr[user_id]
        rated_items_user = self.R.iloc[user_id] > 0
        unrated_items = np.where(rated_items_user == 0)[0]
        predicted_unrated = predicted_ratings_user[unrated_items]
        sorted_unrated_items = unrated_items[np.argsort(predicted_unrated)[::-1]]
        top_recommended_items = sorted_unrated_items[:top_n]
        return top_recommended_items, predicted_unrated[np.argsort(predicted_unrated)[::-1]][:top_n]
