# Pipeline Condition Maintenance System

## Overview
This project focuses on predicting the condition of pipelines using machine learning. By gathering data, and being able to understand how the pipeline works,
we can be able to withstand the damage a bad pipeline can cause

The model classifies pipeline status into:
- Normal
- Moderate
- Critical

The goal is to support predictive maintenance and reduce operational risks in pipeline systems.

## Objectives
- Classify pipeline condition into Normal, Moderate, or Critical
- Identify key features affecting pipeline health
- Build and evaluate multiple classification models using GridSearchCV
- Deploy the model for real-time predictions
- Make the deployment interactive to get an output upon request

## Dataset

The dataset contains pipeline operational data.

### Features:
- Pise Size (mm)
- Thickness (mm)
- Pipe material
- Grade
- Maximum Pressure (psi)
- Temperature (C)
- Corrosion Impact (%)
- Thickness Loss (mm)
- Material Loss (%)
- Time (Years)

### Target Variable:
- Condition (Normal, Moderate, Critical)

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn / XGBoost
- Matplotlib, Seaborn, Plotly
- Mlflow, Joblib
- Streamlit

## Exploratory Data Analysis
- Checked for missing values, there were none
- Carried out a count plot and saw that critical has more data, but the data is not too imbalanced
- Performed correlation analysis
- Did Boxplot to check for outliers, some were found in material loss, but the outliers are explained for
- Visualized feature distributions
- Plot a Normal distribution and noticed that some features were heavily skewed. So, I perfored boxcox transformation for those columns to make the distribution normal.
- It was noticed that there is correlation between pipe size and thickness of pipe. Also, Thickness loss and Material Loss

## Data Preprocessing
- Data cleaning
- Exploratory Data Analysis
- Encoding categorical variables
- Feature scaling
- Train-test split

## Model Building

Models used:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Steps:
- Model training
- Hyperparameter tuning
- Cross-validation
- Handling class imbalance
