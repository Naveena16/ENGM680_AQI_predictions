import pandas as pd
import numpy as np
import glob
import os
import json

def load_air_quality_data(csv_folder_path):
    """
    Load and combine all CSV files from the given folder path into a single DataFrame.
    """
    csv_files_path = glob.glob(os.path.join(csv_folder_path, '*.csv'))
    dataframes = []
    for file_path in csv_files_path:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def clean_data(df):
    """
    Clean the dataset by handling null values and removing invalid rows.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Drop rows with missing values
    df = df.dropna()

    # Remove duplicates if any
    df = df.drop_duplicates()

    # Ensure pollutant values are within realistic ranges
    df = df[(df['PM2.5'] >= 0) & (df['PM2.5'] <= 500)]
    df = df[(df['PM10'] >= 0) & (df['PM10'] <= 604)]
    df = df[(df['SO2'] >= 0) & (df['SO2'] <= 1004)]
    df = df[(df['NO2'] >= 0) & (df['NO2'] <= 2049)]
    df = df[(df['CO'] >= 0) & (df['CO'] <= 50400)]
    df = df[(df['O3'] >= 0) & (df['O3'] <= 1210)]

    print(f"\n\nCombined DataFrame shape after handling nulls and cleaning data: {df.shape}")

    return df

def check_csv_columns(csv_files_path):
    """
    Check if all CSV files in the given folder have the same columns.

    Parameters:
        csv_file_path (List[str]) : Input DataFrame with 'month' and 'hour' columns.
    """
    initial_columns = None
    for file_name in csv_files_path:
        if os.path.isfile(file_name):
            # Load the file into a DataFrame
            df = pd.read_csv(file_name)
            # Get the columns of the current file
            current_columns = list(df.columns)
            if initial_columns is None:
                initial_columns = current_columns
            else:
                if initial_columns == current_columns:
                    continue
                else:
                    raise ValueError(f"Column names are different in file: {file_name}")
    print("All files have the same columns.")


def drop_columns_inplace(dataframe, columns_to_drop):
    """
    Drops specified columns from a DataFrame (inplace).

    Parameters:
        dataframe (pd.DataFrame): The DataFrame from which columns are to be dropped.
        columns_to_drop (list): List of column names to drop.

    Returns:
        None: Modifies the DataFrame in place.
    """
    if not isinstance(columns_to_drop, list):
        raise ValueError("columns_to_drop must be a list of column names.")
    
    dataframe.drop(columns=columns_to_drop, inplace=True, errors='ignore')


def encode_time_features(df):
    """
    Add time-based features such as season and time of day.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with 'month' and 'hour' columns.
        
    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    # Define time of day
    def get_time_of_day(hour):
        if hour < 6:
            return 'Night'
        elif hour < 12:
            return 'Morning'
        elif hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'
    
    df['time_of_day'] = df['hour'].apply(get_time_of_day)
    return df

def calculate_aqi(concentration, breakpoints):
    """
    Calculate AQI for a given pollutant concentration.
    
    Parameters:
        concentration (float): Pollutant concentration.
        breakpoints (list): Breakpoints for AQI calculation.
        
    Returns:
        float: Calculated AQI value.
    """
    for bp in breakpoints:
        C_low, C_high, I_low, I_high = bp
        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
    return None

def calculate_overall_aqi(df, breakpoints_dict):
    """
    Calculate AQI for each pollutant and the overall AQI.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with pollutant columns.
        breakpoints_dict (dict): Dictionary containing breakpoints for each pollutant.
        
    Returns:
        pd.DataFrame: DataFrame with AQI values added.
    """
    for pollutant, breakpoints in breakpoints_dict.items():
        aqi_column = f"{pollutant}_AQI"
        df[aqi_column] = df[pollutant].apply(lambda x: calculate_aqi(x, breakpoints))
    
    # Calculate overall AQI as the maximum AQI across pollutants
    aqi_columns = [f"{pollutant}_AQI" for pollutant in breakpoints_dict.keys()]
    df['AQI'] = df[aqi_columns].max(axis=1)
    return df

def preprocess_data(file_path):
    """
    Full preprocessing pipeline: load, clean, encode features, and calculate AQI.
    
    Parameters:
        file_path (str): Path to the dataset file.
       
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Load data
    df = load_air_quality_data(file_path)
    
    # Clean data
    df = clean_data(df)

    return df


# Function to load breakpoints from the JSON file
def load_breakpoints(json_file_path):
    """
    Loads pollutant-specific breakpoints from a JSON file.
    """
    with open(json_file_path, 'r') as f:
        breakpoints = json.load(f)
    return breakpoints

# Function to calculate AQI for a single pollutant
def calculate_aqi(concentration, breakpoints):
    """
    Calculates AQI for a given pollutant concentration using specified breakpoints.
    """
    for bp in breakpoints:
        C_low, C_high, I_low, I_high = bp
        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
    return None  # If concentration is outside defined range

# Function to add AQI columns for each pollutant and calculate overall AQI
def calculate_aqi_columns(df, json_file_path):
    """
    Adds AQI columns for each pollutant and calculates the overall AQI.
    """
    # Load breakpoints from JSON
    breakpoints = load_breakpoints(json_file_path)
    
    # Add AQI columns to the DataFrame for each pollutant
    for pollutant, bp in breakpoints.items():
        aqi_column = f"{pollutant}_AQI"
        df[aqi_column] = df[pollutant].apply(lambda x: calculate_aqi(x, bp))
    
    # Calculate overall AQI as the maximum of all individual pollutant AQIs
    aqi_columns = [f"{pollutant}_AQI" for pollutant in breakpoints.keys()]
    df['AQI'] = df[aqi_columns].max(axis=1)    
    return df


def generate_alert(aqi):
    if 0 <= aqi <= 50:
        return "Level 1: Excellent - No action needed"
    elif 51 <= aqi <= 100:
        return "Level 2: Good - Sensitive individuals should limit outdoor activities"
    elif 101 <= aqi <= 150:
        return "Level 3: Lightly Polluted - Sensitive Groups reduce outdoor activities"
    elif 151 <= aqi <= 200:
        return "Level 4: Moderately Polluted - Everyone should reduce outdoor exposure"
    elif 201 <= aqi <= 300:
        return "Level 5: Heavily Polluted- Stay indoors and use air purifiers"
    elif aqi > 300:
        return "Level 6: Severely Polluted - Emergency conditions, avoid all outdoor exposure"
    else:
        return "Invalid AQI"
    
def shift_AQI(df):
    """
    Adds columns to the dataset that shift the AQI values by specified intervals to simulate future AQI values.

    Parameters:
    df (DataFrame): Input DataFrame containing the AQI column.

    Returns:
    DataFrame: Modified DataFrame with new AQI columns for future horizons.
    """

    # Print initial data preview for debugging
    print("Initial DataFrame Head:")
    print(df.head(5))
    
    # Add shifted AQI columns for different horizons
    df['AQI_1h'] = df['AQI'].shift(-1)  # 1-hour future AQI
    df['AQI_6h'] = df['AQI'].shift(-6)  # 6-hour future AQI
    df['AQI_24h'] = df['AQI'].shift(-24)  # 24-hour future AQI

    # Drop rows with NaN values resulting from the shift
    df = df.dropna().reset_index(drop=True)

    # Print final data preview for debugging
    print("Modified DataFrame Head:")
    print(df.head(5))
    return df