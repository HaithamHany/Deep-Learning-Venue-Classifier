# Configuration dictionary for hyperparameters
config = {
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "random_state": 42,
    "n_iter": 1,           # Reduced number of iterations
    "cv": 3,               # Number of folds in cross-validation
    "n_jobs": -1,          # Use all available CPU cores
}