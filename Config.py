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
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [16, 32, 64, 128],
    "num_epochs": [10, 50]
}

config_cnn_architecture = {
    "num_layers": 6,
    "filter_sizes": [3, 3, 5, 5, 7, 7],
    "num_filters": [32, 64, 128, 256, 512, 512],
    "strides": [1, 1, 1, 2, 1, 1],
    "paddings": [1, 1, 1, 1, 1, 1],
}