import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def plot_monthly_pollution_levels(air_quality_df):
    """
    Plots monthly average pollution levels for PM2.5, PM10, SO2, NO2, O3, and CO.
    """
    # Set up a 2x2 grid for plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Monthly Average Pollution Levels (PM2.5, PM10, SO2, NO2, O3)
    air_quality_df.groupby('month')[['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']].mean().plot(
        title="Monthly Average Pollution Levels (PM2.5, PM10, SO2, NO2, O3)",
        ax=axs[0, 0]
    )

    # Monthly Average Pollution Levels (CO)
    air_quality_df.groupby('month')[['CO']].mean().plot(
        title="Monthly Average Pollution Levels (CO)",
        ax=axs[0, 1]
    )

    # Hourly Average Pollution Levels (PM2.5, PM10)
    air_quality_df.groupby('hour')[['PM2.5', 'PM10']].mean().plot(
        title="Hourly Average Pollution Levels (PM2.5, PM10)",
        ax=axs[1, 0]
    )

    # Hourly Average Pollution Levels (CO)
    air_quality_df.groupby('hour')[['CO']].mean().plot(
        title="Hourly Average Pollution Levels (CO)",
        ax=axs[1, 1]
    )

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(air_quality_df):
    """
    Plots a heatmap of the correlation matrix for pollutants and environmental factors.
    """
    correlation_matrix = air_quality_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_scatter_relationships(air_quality_df, col1, col2):
    """
    Plots scatter plots to analyze relationships between RAIN/TEMP and PM2.5.
    """
    sns.scatterplot(x=col1, y=col2, data=air_quality_df)
    plt.title(f"Relationship between {col1} and {col2}")
    plt.show()


def plot_average_pollution_by_station(air_quality_df):
    """
    Plots average pollution levels by station for PM2.5 and PM10.
    """
    air_quality_df.groupby('station')[['PM2.5', 'PM10']].mean().plot(kind='bar', title="Average Pollution Levels by Station", figsize=(10, 6))
    plt.xlabel("Station")
    plt.ylabel("Average Pollution Levels")
    plt.tight_layout()
    plt.show()


def plot_histograms(air_quality_df, columns, bins=30):
    """
    Plots histograms for the specified columns.
    """
    air_quality_df[columns].hist(bins=bins, figsize=(12, 6))
    plt.suptitle("Histograms for Selected Columns")
    plt.show()

def plot_boxplots(air_quality_df, columns):
    """
    Plots boxplots for the specified columns.
    """
    air_quality_df[columns].boxplot(figsize=(12, 6))
    plt.title("Boxplots for Selected Columns")
    plt.show()

def apply_log_transformation(air_quality_df, columns):
    """
    Applies log transformation to the specified columns and adds new columns to the DataFrame.
    Returns the updated DataFrame.
    """
    for col in columns:
        log_col_name = f"{col}_log"
        air_quality_df[log_col_name] = np.log1p(air_quality_df[col])
    return air_quality_df

def plot_transformed_boxplots(air_quality_df, transformed_columns):
    """
    Plots boxplots for log-transformed columns.
    """
    plt.figure(figsize=(10, 6))
    air_quality_df[transformed_columns].boxplot()
    plt.title("Boxplot after Log Transformation")
    plt.show()


def plot_average_aqi_by_season(df):
    """
    Plots the average AQI by season as a bar chart.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='AQI', data=df, ci=None, palette='Set2')
    plt.title('Average AQI by Season')
    plt.xlabel('Season')
    plt.ylabel('Average AQI')
    plt.show()

def plot_average_aqi_by_time_of_day(df):
    """
    Plots the average AQI by time of day as a bar chart.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='time_of_day', y='AQI', data=df, ci=None, palette='Set3')
    plt.title('Average AQI by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Average AQI')
    plt.show()

def plot_heatmap_aqi_season_time_of_day(df):
    """
    Plots a heatmap of average AQI by season and time of day.
    """
    heatmap_data = df.pivot_table(
        values='AQI',
        index='season',
        columns='time_of_day',
        aggfunc='mean'
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", cbar=True)
    plt.title('Heatmap of Average AQI by Season and Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Season')
    plt.show()

def plot_density_aqi_by_season(df):
    """
    Plots a density plot of AQI by season.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='AQI', hue='season', fill=True, alpha=0.5, palette='coolwarm')
    plt.title('Density Plot of AQI by Season')
    plt.xlabel('AQI')
    plt.ylabel('Density')
    plt.show()

def plot_density_aqi_by_time_of_day(df):
    """
    Plots a density plot of AQI by time of day.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='AQI', hue='time_of_day', fill=True, alpha=0.5, palette='viridis')
    plt.title('Density Plot of AQI by Time of Day')
    plt.xlabel('AQI')
    plt.ylabel('Density')
    plt.show()



def feature_importance_mutual_info(df, selected_features, target):
    """
    Calculate and plot feature importance based on mutual information regression.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - selected_features: list of features to analyze
    - target: the target variable for prediction
    
    Returns:
    - A pandas DataFrame containing mutual information scores for features
    """
    # Extract features and target
    X = df[selected_features]
    y = df[target]
    
    # Compute mutual information scores
    mi_scores = mutual_info_regression(X, y)
    
    # Create a DataFrame for feature importance
    mi_scores_df = pd.DataFrame({
        'Feature': selected_features,
        'Mutual Information': mi_scores
    }).sort_values(by='Mutual Information', ascending=False)
    
    # Display feature importance
    print("Feature Importance based on Mutual Information:")
    print(mi_scores_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(mi_scores_df['Feature'], mi_scores_df['Mutual Information'], color='skyblue')
    plt.xlabel('Mutual Information')
    plt.ylabel('Features')
    plt.title('Feature Importance for AQI Prediction')
    plt.gca().invert_yaxis()
    plt.show()
    
    return mi_scores_df