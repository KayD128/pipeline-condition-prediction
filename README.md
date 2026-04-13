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
- KNN Classifier
- SVC (Support Vector Classifier) 
- Decision Tree
- Random Forest
- XGBoost

### Steps:
- Model training
- Hyperparameter tuning using GridSearchCV
- Comparing metrics
- Selection of the Best metrics
- Using Pipeline to bring the preprocessing and modelling together in a model
- Cross-validation
- Joblib dump

## Model Evaluation

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

| Model           | Accuracy |
|-----------------|----------|
| Decision Tree         | 98%      |
| Random Forest        | 98%      |
| SVC       | 97.5%      |
| KNN        | 68%      |
| Logistic Regression        | 96.5%      |

### I ended up using Random Forest Classifier for the work with hyperparameters:
-  class_weight = None
-  random_state = 42
-  max_depth = 4
-  min_samples_leaf = 4
-  n_estimators = 100

## Key Insights
- Feature X strongly influences Critical condition
- Thickness Loss greatly affects the Target variable
- Moderate class has more data than the others
- Model performs best with Random Forest Classifier
- The cross validation shows how well the data performs. 3 out of 5 folds gives 0.99

## Deployment

The model is deployed using Streamlit on Render as host.

### Run Locally:
git clone https://github.com/KayD128/pipeline-condition-prediction.git
cd root folder
pip install -r requirements.txt
streamlit run appipe.py

### Live App:
https://pipeline-condition-prediction.onrender.com/

## Future Improvements
- Add real-time data integration
- Deploy with Docker
- Bring in more data
- Create a Dashboard with said Data

## License
MIT License
