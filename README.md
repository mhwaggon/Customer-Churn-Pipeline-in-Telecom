# Customer Churn Prediction Model

## Overview

This project builds a machine learning model to predict customer churn. The goal is to identify customers who are likely to stop using a service so that the business can take proactive retention actions.

The model uses historical customer data to learn behavioral patterns associated with churn and generates predictions that can support marketing, customer success, and revenue protection strategies.

## Business Objective

Customer acquisition is expensive, so retaining existing customers is critical. This project helps answer:

* Which customers are most likely to churn?
* What factors are most strongly associated with churn?
* How can the business intervene before churn occurs?

The output of this model can be used to prioritize retention campaigns, offer targeted incentives, and reduce revenue loss.

## Dataset

The dataset contains customer-level information such as:

* Demographics
* Account details
* Service usage
* Contract type
* Billing information
* Customer tenure
* Churn indicator (target variable)

Target variable:

* Churn (binary: Yes or No)

## Project Workflow

### 1. Data Preprocessing

* Removed missing or inconsistent values
* Encoded categorical variables
* Scaled numerical features where appropriate
* Converted churn label to binary format
* Split data into training and testing sets

### 2. Exploratory Data Analysis

* Analyzed churn distribution
* Examined relationships between churn and key features
* Identified high-risk customer segments
* Evaluated feature correlations

### 3. Feature Engineering

* Transformed categorical variables
* Created derived features where useful
* Standardized numeric inputs for model stability

### 4. Model Development

Multiple machine learning models were evaluated, such as:

* Logistic Regression
* Random Forest
* Gradient Boosting / XGBoost (if used)
* Other classification algorithms

Models were compared using performance metrics on validation data.

### 5. Model Evaluation

Key evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Because churn is often imbalanced, special focus was placed on recall and F1 score to correctly identify customers at risk.

### 6. Model Selection

The best performing model was selected based on its ability to detect churn while maintaining balanced overall performance.

## Results

The final model successfully identifies customers with elevated churn risk. Feature importance analysis revealed that factors such as contract type, tenure, and service usage strongly influence churn behavior.

These insights can guide targeted retention strategies.

## How to Use the Model

1. Provide customer feature data in the same format used during training.
2. Run the prediction pipeline.
3. Review churn probability scores.
4. Target high-risk customers with retention actions.


