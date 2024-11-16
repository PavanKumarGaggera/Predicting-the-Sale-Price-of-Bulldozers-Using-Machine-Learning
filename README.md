
# Predicting the Sale Price of Bulldozers Using Machine Learning

## Project Overview
This project aims to develop a machine learning model to predict the sale price of bulldozers based on historical data. Using data from the Kaggle Bluebook for Bulldozers competition, this project demonstrates an end-to-end machine learning workflow, including data cleaning, feature engineering, model selection, and evaluation.

## Problem Definition
Predict the future sale price of a bulldozer given its characteristics and historical auction sales data. This problem focuses on building a regression model that minimizes the Root Mean Squared Logarithmic Error (RMSLE) between predicted and actual prices.

## Dataset
The dataset used for this project is sourced from the Kaggle Bluebook for Bulldozers competition, and it includes three main files:
- **Train.csv**: Training data (data through the end of 2011).
- **Valid.csv**: Validation data (data from January 1, 2012, to April 30, 2012).

You can find more details about the competition and download the dataset [here](https://www.kaggle.com/c/bluebook-for-bulldozers).

## Evaluation Metric
The primary evaluation metric for this competition is the **Root Mean Squared Log Error (RMSLE)** between the predicted and actual bulldozer prices.

## Features
The dataset contains numerous features describing different aspects of the bulldozers. A comprehensive data dictionary detailing all features is available [here](https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing).

## Project Steps
### 1. Data Preprocessing & Cleaning
- Handled missing values.
- Transformed date-related fields to extract useful features (e.g., year sold).
- Encoded categorical variables.

### 2. Exploratory Data Analysis (EDA)
- Visualized key data patterns and distributions.
- Identified correlations and feature importance for prediction.

### 3. Feature Engineering
- Created new features based on domain knowledge.
- Converted date features and handled categorical variables effectively.

### 4. Model Building & Selection
- Implemented multiple machine learning models including Random Forest and XGBoost regressors.
- Conducted hyperparameter tuning for model optimization.

### 5. Model Evaluation
- Evaluated models using the RMSLE metric on validation data.
- Selected the best-performing model for further analysis.

## Key Results
- Achieved an RMSLE score of 0.24 with the best-performing model.
- Demonstrated the impact of feature engineering and hyperparameter tuning on model performance.

## How to Use
### Prerequisites
Make sure you have Python 3.x installed, along with the following libraries:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bulldozer-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd bulldozer-price-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook bulldozer_price_prediction.ipynb
   ```


## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost

## Acknowledgments
- Kaggle Bluebook for Bulldozers competition for providing the dataset and challenge.
- Open-source libraries and community tutorials that supported this project.
