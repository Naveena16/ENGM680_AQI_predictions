import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_vif(X):
    """
    Calculates Variance Inflation Factor (VIF) for a given DataFrame of features.
    Returns a DataFrame with features and their VIF scores.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def plot_random_forest_feature_importance(X, y, random_state=42):
    """
    Fits a Random Forest model and plots feature importance.
    """
    rf_model = RandomForestRegressor(random_state=random_state)
    rf_model.fit(X, y)
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importance from Random Forest")
    plt.show()