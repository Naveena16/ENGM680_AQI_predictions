import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def fit_ols_regression(X, y):
    """
    Fits an OLS regression model and returns the model summary.
    """
    X = sm.add_constant(X)  # Add constant for regression
    model = sm.OLS(y, X).fit()
    return model.summary()

def eval_final_model(X,y, best_params, test_size=0.2, random_state=42):
    """
    Train and evaluate the final Random Forest model using the best parameters from grid search.

    Parameters:
    - X: Features (DataFrame or numpy array)
    - y: Target variable (Series or numpy array)
    - best_params: Dictionary of the best hyperparameters from grid search
    - test_size: Test split proportion (default=0.2)
    - random_state: Random state for reproducibility (default=42)

    Returns:
    - metrics: Dictionary containing MAE, RMSE, and R² Score
    - final_model: Trained Random Forest model
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    # Initialize the model with the best parameters
    final_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=random_state
    )
    
    # Train the model
    final_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = final_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = final_model.score(X_test, y_test)
    
    metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Squared Error (RMSE)": rmse,
        "R² Score": r2
    }
    
    return metrics, final_model

