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

# Configuration dictionary for CNN hyperparameters
config_cnn = {
    "learning_rate": [0.001, 0.0001],
    "batch_size": [32, 64],
    "num_epochs": [4, 10],
}

config_cnn_architecture = {
    "num_layers": 4,
    "filter_sizes": [3, 3, 3, 3],
    "num_filters": [32, 64, 128, 256],
    "strides": [1, 1, 1, 1],
    "paddings": [1, 1, 1, 1],
}