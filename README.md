# Early Prediction of Sepsis from Clinical Data

## Overview

Sepsis is a critical medical condition that requires timely detection to improve patient outcomes. This project applies **machine learning** methods to analyze clinical data and predict the onset of sepsis. By addressing challenges such as missing data, temporal dependencies, and imbalanced classes, the project demonstrates the application of data science techniques to a real-world healthcare problem.

This project is adapted from the dataset and methodology described in the publication:\
[*"Data-DrivenApproaches for Early Sepsis Detection"*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8193357/).\
The work builds on the provided data and concepts, applying additional steps for preprocessing, feature engineering, and modeling.

### Features

-   **Exploratory Data Analysis (EDA)** to assess data quality and identify patterns.

-   **Data preprocessing** methods for handling missing values and preparing time-series data.

-   **Feature engineering** to incorporate temporal aspects and enrich predictive power.

-   Implementation of **machine learning models**, including Random Forest and XGBoost.

-   Analysis of model performance and feature importance for interpretability.

-   Sub-population analysis to evaluate model robustness across different patient groups.

### Results

-   The **baseline model** (Random Forest) achieved an F1-score of **0.7** on the test set, demonstrating the ability to identify sepsis effectively.

-   The **advanced model** (XGBoost) improved recall and achieved an F1-score of **0.758**, indicating a better balance between precision and recall.

-   Sub-population analysis highlighted robust performance:

    -   Patients with the most missing data: F1-score of **0.972**.

    -   Senior patients (75+ years): F1-score of **0.798**.

-   Demonstrated the importance of missing data indicators as predictive features and the value of temporal aggregation for modeling.

## Machine Learning Pipeline

### 1. **Data Preprocessing**

-   Handled missing values through transformations, interpolation, and imputation.

-   Generated additional features to indicate the presence of missing data.

-   Aggregated temporal data into feature vectors representing patient health over time.

### 2. **Feature Engineering**

-   Extracted features capturing trends in patient conditions over multiple time points.

-   Calculated statistical summaries to enhance the representation of patient data.

### 3. **Modeling**

-   Trained and evaluated machine learning models:

    -   **Baseline Model**: Random Forest, optimized with hyperparameter tuning.

    -   **Advanced Model**: XGBoost, incorporating gradient boosting and regularization.

-   Assessed model performance using precision, recall, and F1-score to ensure balanced predictions.

### 4. **Interpretability**

-   Used feature importance metrics to identify the most predictive variables.

-   Conducted sub-population analysis to understand model performance across different patient demographics.
