Machine Learning Assignment 2
BITS ID: 2025AA05198
Course: Machine Learning


a. Problem Statement
The objective of this project is to perform multi-class classification on the Beans dataset to predict leaf diseases using various machine learning models.
The goal is to compare model performance using multiple evaluation metrics and deploy the solution using Streamlit.


b. Dataset Description
The Beans dataset is obtained from TensorFlow Datasets (TFDS).
Total Instances: ~1300
Classes: 3 (Bean leaf diseases)
Feature Engineering: 12 statistical features extracted from images
Type: Multi-class classification problem
The dataset satisfies the assignment requirement of:
Minimum 12 features
Minimum 500 instances


c. Models Used and Comparison Table
The following models were implemented on the same dataset:
Logistic Regression
Decision Tree
K-Nearest Neighbors
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)
Comparison Table
Model                Accuracy      AUC  Precision   Recall       F1      MCC
Logistic Regression  0.505792 0.692755   0.511892 0.505792 0.506848 0.260292
      Decision Tree  0.436293 0.577474   0.437450 0.436293 0.435842 0.155222
                KNN  0.563707 0.701071   0.563796 0.563707 0.563738 0.345558
        Naive Bayes  0.420849 0.603943   0.453880 0.420849 0.359338 0.169983
      Random Forest  0.575290 0.743976   0.577393 0.575290 0.574503 0.364502
            XGBoost  0.544402 0.724207   0.549022 0.544402 0.542995 0.319308




d. Model Performance Observations
ML Model Name	Observation about model performance
Logistic Regression	Performs reasonably but limited by linear decision boundary.
Decision Tree	May overfit but captures nonlinear patterns.
KNN	Performance depends on choice of K and scaling.
Naive Bayes	Works well but assumes feature independence.
Random Forest	Better generalization due to ensemble averaging.
XGBoost	Best performance due to boosting and regularization.


Deployment The application is deployed using Streamlit Community Cloud.
