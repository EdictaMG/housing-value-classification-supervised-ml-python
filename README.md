# Predicting Housing Value Using Supervised Machine Learning with Python

In this classification project, I built a supervised machine learning model to predict whether a home is expensive or not based on its features.

## Situation

Using a dataset of historical home prices in Ames, Iowa, the goal was to classify homes as either "expensive" or "not expensive" based on features such as property size, number of bedrooms, presence of a pool, and more.

This project was part of a classification competition, where models were tested against actual labels on an online platform. The final model I submitted achieved an accuracy of **0.9788**.

## Approach

To begin, I researched the domain to better understand which housing features typically influence price. I then performed exploratory data analysis to investigate relationships in the dataset.

After preprocessing the data, I trained and evaluated several classification models using Scikit-learn, including:

- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Logistic Regression  
- Random Forest  
- XGBoost  

The Random Forest model, fine-tuned with GridSearchCV for hyperparameter optimization, delivered the best results in terms of accuracy and generalization. I tested removing several columns, but this had minimal impact on model performance—apart from the ID column, which was excluded as it held no predictive value.

## Files

### Data
- `housing-classification-train.csv`: training data  
- `housing-classification-test.csv`: test data  

### Script
- `Classification_ML_random_forest.ipynb`: Notebook documenting the full process using the Random Forest model

### Document
- `housing_classification_features_descriptions.txt`: Describes each feature in the dataset for easier interpretation and analysis.

## Using the Files

1. Download the data files and notebook and save them to your Google Drive.  
2. Update the file paths in the notebook as needed.  
3. Run the notebook in Google Colab or Jupyter to reproduce the analysis.

## Languages and Libraries

- Python 3.10.12  
- Pandas 2.2.2  
- NumPy 1.26.4  
- scikit-learn 1.4.2  

## Tools

- Google Colab (or Jupyter Notebook)  
- Google Drive for storage
