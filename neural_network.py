"""
Neural Network Regression with GridSearchCV
This script performs hyperparameter tuning for neural network regression.
Run with: python neural_network.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variables to prevent TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure GPU memory growth to prevent memory allocation issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass  # Memory growth must be set before GPUs are initialized

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress TensorFlow info messages
tf.get_logger().setLevel('ERROR')

# Enable mixed precision for faster training on compatible GPUs (RTX 20xx+, A100, etc.)
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    MIXED_PRECISION_ENABLED = True
except Exception:
    MIXED_PRECISION_ENABLED = False

print("=" * 70)
print("NEURAL NETWORK REGRESSION WITH GRIDSEARCHCV")
print("=" * 70)

# Check GPU availability
print("\n" + "=" * 70)
print("GPU/CUDA Availability Check")
print("=" * 70)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  CUDA available: {len(gpus)} GPU(s) detected")
    for gpu in gpus:
        print(f"    - {gpu}")
    if MIXED_PRECISION_ENABLED:
        print(f"  Mixed precision (float16): ENABLED")
else:
    print("  No GPU detected. Training will use CPU.")
print("=" * 70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading data from parquet files...")

X_train = pd.read_parquet('X_train_scaled_df.parquet')
X_test = pd.read_parquet('X_test_scaled_df.parquet')
y_train = pd.read_parquet('y_train.parquet').values
y_test = pd.read_parquet('y_test.parquet').values

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape:  {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape:  {y_test.shape}")

# ============================================================================
# STEP 2: Define Neural Network Model Builder
# ============================================================================
print("\n[STEP 2] Defining neural network architecture...")

def create_nn_model(n_units=64, n_layers=2, dropout_rate=0.2, l2_reg=0.01, learning_rate=0.001, meta=None):
    """
    Create a neural network model with configurable architecture.

    Parameters:
    -----------
    n_units : int
        Number of units in the first hidden layer
    n_layers : int
        Total number of hidden layers
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization factor
    learning_rate : float
        Learning rate for Adam optimizer
    meta : dict
        Metadata from scikeras (contains X shape)

    Returns:
    --------
    model : Sequential
        Compiled Keras model
    """
    model = Sequential()

    # Get input dimension from meta if available, else use default
    if meta is not None and 'X_in_' in meta:
        input_dim = meta['X_in_'].shape[1]
    else:
        input_dim = X_train.shape[1]

    # Input layer
    model.add(Dense(
        n_units,
        input_dim=input_dim,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    ))
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for i in range(n_layers - 1):
        model.add(Dense(
            max(n_units // (i + 1), 16),  # Decreasing units, minimum 16
            activation='relu',
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model

print(f"  Input dimension: {X_train.shape[1]}")
print("  Model function defined")

# ============================================================================
# STEP 3: Create KerasRegressor Wrapper
# ============================================================================
print("\n[STEP 3] Creating KerasRegressor wrapper...")

nn_model = KerasRegressor(
    model=create_nn_model,
    verbose=0,
    epochs=100,
    batch_size=32,
    validation_split=0.1
)

# Create pipeline
pipeline_nn = Pipeline([
    ('regressor', nn_model)
])

print("  Pipeline created")

# ============================================================================
# STEP 4: Define Hyperparameter Grid
# ============================================================================
print("\n[STEP 4] Defining hyperparameter grid...")

param_grid_nn = {
    'regressor__model__n_units': [64, 128, 256],
    'regressor__model__n_layers': [2, 3, 4],
    'regressor__model__dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'regressor__model__l2_reg': [0.001, 0.01, 0.1],
    'regressor__model__learning_rate': [0.0001, 0.001, 0.01],
    'regressor__batch_size': [32, 64, 128],
    'regressor__epochs': [100, 150]
}

# Calculate total combinations
total_combinations = (
    len(param_grid_nn['regressor__model__n_units']) *
    len(param_grid_nn['regressor__model__n_layers']) *
    len(param_grid_nn['regressor__model__dropout_rate']) *
    len(param_grid_nn['regressor__model__l2_reg']) *
    len(param_grid_nn['regressor__model__learning_rate']) *
    len(param_grid_nn['regressor__batch_size']) *
    len(param_grid_nn['regressor__epochs'])
)

print(f"  Total hyperparameter combinations: {total_combinations}")
print(f"  With 5-fold CV: {total_combinations * 5} model fits")

# ============================================================================
# STEP 5: Create Custom MAPE Scorer
# ============================================================================
print("\n[STEP 5] Creating custom MAPE scorer...")

mape_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False
)

print("  MAPE scorer created (lower is better)")

# ============================================================================
# STEP 6: Perform GridSearchCV
# ============================================================================
print("\n[STEP 6] Starting GridSearchCV...")
print("  This may take a while...")
print("-" * 70)

grid_search_nn = GridSearchCV(
    pipeline_nn,
    param_grid_nn,
    cv=5,
    scoring=mape_scorer,
    n_jobs=1,  # n_jobs=1 for TensorFlow to avoid threading issues
    verbose=2,
    return_train_score=True
)

# Fit the model
grid_search_nn.fit(X_train, y_train)

print("-" * 70)
print("GridSearchCV completed!")

# ============================================================================
# STEP 7: Get Best Model and Parameters
# ============================================================================
print("\n[STEP 7] Retrieving best model...")

best_nn_model = grid_search_nn.best_estimator_
best_nn_params = grid_search_nn.best_params_
best_nn_mape = -grid_search_nn.best_score_

print(f"\n  Best Neural Network cross-validation MAPE: {best_nn_mape:.4f}%")
print("\n  Best Neural Network parameters:")
for param, value in best_nn_params.items():
    print(f"    {param}: {value}")

# ============================================================================
# STEP 8: Evaluate on Test Set
# ============================================================================
print("\n[STEP 8] Evaluating on test set...")

nn_test_pred = best_nn_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, nn_test_pred)
r2 = r2_score(y_test, nn_test_pred)
mae = mean_absolute_error(y_test, nn_test_pred)
rmse = np.sqrt(mse)

# Handle MAPE calculation for potential zero values
epsilon = 1e-10
non_zero_idx = np.abs(y_test) > epsilon
mape = np.mean(np.abs((y_test[non_zero_idx] - nn_test_pred[non_zero_idx]) / y_test[non_zero_idx])) * 100

# Adjusted R2
n = X_test.shape[1]  # number of features
p = len(y_test)      # number of samples
adj_r2 = 1 - (1 - r2) * (p - 1) / (p - n - 1)

print(f"\n  Test Set Performance:")
print(f"    MSE:        {mse:.4f}")
print(f"    RMSE:       {rmse:.4f}")
print(f"    MAE:        {mae:.4f}")
print(f"    MAPE:       {mape:.4f}%")
print(f"    R²:         {r2:.4f}")
print(f"    Adj. R²:    {adj_r2:.4f}")

# ============================================================================
# STEP 9: Save Best Model and Parameters
# ============================================================================
print("\n[STEP 9] Saving best model...")

# Save the best model
joblib.dump(best_nn_model, 'best_nn.joblib')
print("  Best model saved to: best_nn.joblib")

# Save the best parameters
joblib.dump(best_nn_params, 'best_nn_params.joblib')
print("  Best parameters saved to: best_nn_params.joblib")

# Save grid search results
results_df = pd.DataFrame(grid_search_nn.cv_results_)
results_df.to_parquet('nn_gridsearch_results.parquet', index=False)
print("  Grid search results saved to: nn_gridsearch_results.parquet")

# ============================================================================
# STEP 10: Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nNeural Network Regression Results:")
print(f"  Best CV MAPE:  {best_nn_mape:.4f}%")
print(f"  Test MAPE:     {mape:.4f}%")
print(f"  Test R²:       {r2:.4f}")
print(f"\nBest Hyperparameters:")
for param, value in best_nn_params.items():
    print(f"  {param}: {value}")

print("\nFiles created:")
print("  - best_nn.joblib              (trained model)")
print("  - best_nn_params.joblib       (best hyperparameters)")
print("  - nn_gridsearch_results.parquet (full grid search results)")

print("\n" + "=" * 70)
print("NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
