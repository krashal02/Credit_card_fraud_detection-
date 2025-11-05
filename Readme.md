# Credit Card Fraud Detection

This notebook demonstrates a machine learning approach to detect fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset, undersampling is used to balance the classes before training a logistic regression model.

## Dataset

The dataset used in this notebook is the Credit Card Fraud Detection dataset, which contains anonymized credit card transactions. The dataset is highly imbalanced, with a very small percentage of transactions being fraudulent.
Dataset Link :- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Approach

1.  **Data Loading and Exploration:** Load the dataset and perform initial exploration to understand the data distribution, including the class distribution.
2.  **Handling Imbalanced Data:** Employ undersampling to create a balanced dataset by randomly sampling the majority class (non-fraudulent transactions) to match the number of minority class (fraudulent transactions).
3.  **Splitting Data:** Split the balanced dataset into training and testing sets.
4.  **Model Training:** Train a Logistic Regression model on the training data.
5.  **Model Evaluation:** Evaluate the trained model's performance on both the training and testing data using accuracy score.

## Code

The notebook contains the following key steps:

-   Loading the dataset using pandas.
-   Checking the class distribution.
-   Separating legitimate and fraudulent transactions.
-   Performing undersampling on the legitimate transactions to match the number of fraudulent transactions.
-   Concatenating the sampled legitimate transactions with the fraudulent transactions to create a new, balanced dataset.
-   Splitting the balanced data into features (X) and target (y).
-   Splitting the data into training and testing sets using `train_test_split` with stratification.
-   Initializing and training a `LogisticRegression` model.
-   Predicting on the training and testing sets.
-   Calculating and printing the accuracy scores for both training and testing data.

## How to use

1.  Open the notebook in Google Colab or any Jupyter Notebook environment.
2.  Make sure you have the `creditcard.csv` file available.
3.  Run the cells sequentially to execute the data loading, processing, model training, and evaluation steps.

## Dependencies

-   pandas
-   numpy
-   sklearn

These libraries are commonly available in Colab and other Python environments.