import math
import numpy as np
from get_items_rated import get_items_rated_by_user

def evaluate_RMSE(Yhat, rates, n_users, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred 
        se += (e*e).sum(axis = 0)
        cnt += e.size 
    return math.sqrt(se/cnt)

def evaluate_MAE(Yhat, rates, n_users, W, b):
    abs_error_sum = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        abs_e = abs(scores_truth - scores_pred)
        abs_error_sum += abs_e.sum(axis = 0)
        cnt += abs_e.size 
    return abs_error_sum / cnt

def evaluate_precision_at_k(Yhat, rates, n_users, W, b, k = 5):
    total_precision = 0.0
    
    # Loop over all users
    for n in range(n_users):
        # Get the rated items and the true ratings for user n
        ids, scores_truth = get_items_rated_by_user(rates, n)
        
        # Get the predicted scores for these items for user n
        scores_pred = Yhat[ids, n]
        
        # Sort the items by predicted score in descending order and get the top k indices
        top_k_indices = np.argsort(scores_pred)[-k:]  # Indices of the top k items
        
        # Check how many of the top k items are actually relevant (i.e., rated by the user)
        relevant_recommended = sum(1 for idx in top_k_indices if idx in ids)
        
        # Calculate Precision@KPrecision@K for this user
        precision_k = relevant_recommended / k
        total_precision += precision_k
    
    # Return the average Precision@K over all users
    return total_precision / n_users