# Air Quality Prediction Project

## Overview

This project focuses on predicting Air Quality Index (AQI) based on pollutant levels and environmental factors. It involves the following key steps:

1. **Data Cleaning and Preprocessing**  
   Handling missing values, outliers, and transforming the dataset into a clean format.

2. **Exploratory Data Analysis (EDA)**  
   Visualizing patterns and relationships between pollutants and environmental variables.

3. **Feature Engineering**  
   Selecting and transforming features for model training.

4. **Model Training and Evaluation**  
   Training various models, hyperparameter tuning, and evaluating their performance.

5. **Future Prediction**  
   Simulating AQI prediction for future time intervals (e.g., 1-hour, 6-hour, 24-hour).

6. **Alert System**  
   Developing an alert mechanism to warn users about poor air quality levels.

## Project Structure

```
├── airquality_dataset       # Dataset for different stations
├── README.md                # Overview and instructions
├── src/                     # Core project code
│   ├── __init__.py
│   ├── data_processing.py   # Data cleaning and preprocessing
│   ├── eda.py               # Exploratory Data Analysis functions
│   ├── feature_engineering.py # Feature engineering and selection
│   ├── model_training.py    # Model training, hyperparameter tuning, and future prediction
│   ├── model_evaluation.py  # Model evaluation metrics and reporting
├── requirements.txt         # List of required Python packages
├── final_pipeline.ipynb     # Final pipeline notebook for running the project
├── breakpoints.json         # AQI breakpoints for pollutant calculations
└── modelcard.pdf            # Model card with project insights and summary
```

## Steps to Run the Project

1. **Create a Virtual Environment:**
   - Ensure you have Python 3.8 or higher installed.
   - Create a virtual environment using the following command:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       .\venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

2. **Install Dependencies:**
   - Use the following command to install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Dataset:**
   - Ensure the dataset is located in the `airquality_dataset` directory.
   - If using a different dataset path, update the path in `config.yaml` accordingly.

4. **Run the Project:**
   - Open the `final_pipeline.ipynb` notebook located in the `main` directory.
   - Execute the notebook to run the full workflow.

## Future Enhancements

- **Adding support for multiple datasets from different regions**: Extend the project to handle and analyze air quality datasets from various geographical locations, enabling broader applicability and insights.

- **Optimizing models for real-time AQI predictions**: Improve model efficiency and performance to support real-time air quality predictions, ensuring timely alerts and decision-making.

- **Deploying a web-based dashboard for visualizations and alerts**: Develop and deploy an interactive web-based platform for users to view visualizations, monitor AQI trends, and receive real-time alerts about air quality.
