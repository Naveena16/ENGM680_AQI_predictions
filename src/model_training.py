from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def train_and_evaluate_linear_regression(X, y, test_size, random_state):
    """
    Train and evaluate a Linear Regression model for predicting the target variable.

    Parameters:
    - df: pandas DataFrame containing the dataset
    - features: list of feature column names
    - target: target variable column name
    - test_size: proportion of the dataset to include in the test split (default=0.2)
    - random_state: seed used by the random number generator (default=42)

    Returns:
    - metrics: dictionary containing MAE and RMSE
    - model: trained Linear Regression model
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Initialize and train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Print evaluation metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Return metrics and model
    return {"MAE": mae, "RMSE": rmse}, lr_model
    

def train_random_forest(X, y, test_size=0.2, random_state=42):
    """
    Train a Random Forest Regressor and evaluate its performance.

    Parameters:
    - X: Features (DataFrame or numpy array)
    - y: Target variable (Series or numpy array)
    - test_size: Test split proportion (default=0.2)
    - random_state: Random state for reproducibility (default=42)

    Returns:
    - metrics: Dictionary containing MAE and RMSE
    - model: Trained Random Forest model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    
    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=False))
    
    return {"MAE": mae, "RMSE": rmse}, rf_model

def train_gradient_boosting(X, y, test_size, random_state):
    """
    Train a Gradient Boosting Regressor and evaluate its performance.

    Parameters:
    - X: Features (DataFrame or numpy array)
    - y: Target variable (Series or numpy array)
    - test_size: Test split proportion (default=0.2)
    - random_state: Random state for reproducibility (default=42)

    Returns:
    - metrics: Dictionary containing R2 Score and RMSE
    - model: Trained Gradient Boosting model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    
    # Train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(random_state=random_state)
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gb_model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return {"R2 Score": r2, "RMSE": rmse}, gb_model

def perform_grid_search(X, y, param_grid, test_size, random_state):
    """
    Perform hyperparameter tuning using Grid Search for Random Forest.

    Parameters:
    - X: Features (DataFrame or numpy array)
    - y: Target variable (Series or numpy array)
    - param_grid: Dictionary containing hyperparameters for tuning
    - test_size: Test split proportion (default=0.2)
    - random_state: Random state for reproducibility (default=42)

    Returns:
    - best_params: Best parameters found by Grid Search
    - best_score: Best score achieved
    - grid_search: Fitted GridSearchCV object
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    
    # Perform Grid Search
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=random_state),
                               param_grid=param_grid,
                               cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_score_, grid_search


def train_models_for_horizons(df, features, horizons, test_size, random_state):
    """
    Train models for predicting AQI at different horizons.

    Parameters:
    df (pd.DataFrame): DataFrame containing features and target horizons.
    features (list): List of feature column names.
    horizons (list): List of target horizon column names (e.g., ['AQI_1h', 'AQI_6h', 'AQI_24h']).
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    dict: Trained models for each horizon.
    dict: Evaluation metrics for each horizon.
    """
    models = {}
    metrics = {}

    for horizon in horizons:
        print(f"\nTraining for Horizon: {horizon}")
        
        # Features and target
        X = df[features]
        y = df[horizon]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train the model (e.g., Random Forest)
        rf_model = RandomForestRegressor(random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        
        # Evaluate performance
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = rf_model.score(X_test, y_test)
        
        metrics[horizon] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        }
        
        models[horizon] = rf_model
        print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    
    return models, metrics
