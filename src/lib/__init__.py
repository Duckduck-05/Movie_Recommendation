# src/lib/__init__.py

# This file makes the src/lib directory a Python package.
# You can include any package-level imports or initializations here if needed.

# Example: Importing functions from the modules
from .collaborative_filtering import CF
from .evaluate import evaluate_MAE, evaluate_RMSE
from .get_items_rated import get_items_rated_by_user