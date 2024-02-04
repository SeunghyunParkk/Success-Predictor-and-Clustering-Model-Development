# Kickstarter Success Predictor & Developing a Clustering Model

## Project Overview
This project aims to predict the success of Kickstarter campaigns and develop a clustering model using machine learning models. The dataset includes a variety of features related to the campaign, which undergo preprocessing, feature selection, and finally, prediction using models like Gradient Boosting and Random Forest.

## Installation
To run this project, you will need to install the following Python libraries:
- pandas
- sklearn
- numpy
- statsmodels

You can install these packages using pip:
```bash
pip install pandas scikit-learn numpy statsmodels

## Dataset
The code requires two Excel datasets named Kickstarter.xlsx and Kickstarter-Grading-Sample.xlsx. Ensure these files are in the same directory as the script or update the file paths in the script accordingly.

# Kickstarter Success Predictor

## Overview of the Script

### Data Preprocessing

The dataset used is Kickstarter.xlsx, which includes various details about Kickstarter projects. The preprocessing steps include:

* Loading the data using pandas.
* Dropping irrelevant columns and rows with missing values.
* Encoding categorical variables into numerical values.
* Standardizing the features for machine learning analysis.

### Feature Selection

Two feature selection methods are employed and their hyperparameters are tuned using GridSeachCV:

1. **LASSO (Least Absolute Shrinkage and Selection Operator):** Identifies significant features by applying L1 regularization.
2. **Random Forest:** Utilizes the feature importance scores to select relevant features.

### Hyperparameter Tuning and Model Evaluation

The project explores two main models with hyperparameters tuned using GridSearchCV:

* Random Forest Classifier: Tuned for optimal hyperparameters and feature sets.
* Gradient Boosting Classifier: Similarly tuned for best performance.
The models are evaluated based on accuracy, and the best-performing model is selected for the final prediction.

### Final Model Selection

In conclusion, the Gradient Boosting model, coupled with features selected by the Random Forest method, is emerged as the most effective, achieving the highest accuracy of 74.6%. 

## Out-of-Sample Validation

Validation is performed using a separate dataset, Kickstarter-Grading-Sample.xlsx, to ensure the model's generalizability.

## Results

The final Gradient Boosting model demonstrated promising results in predicting the success of Kickstarter projects. Additionally, the clustering analysis provided insightful patterns, highlighting the diversity of projects and their characteristics.

## Conclusion

This project illustrates the potential of machine learning in predicting crowdfunding success and offers valuable insights into the factors that contribute to a project's outcome. It also demonstrates the effectiveness of clustering analysis in uncovering underlying patterns in complex datasets.




# Clustering Model Development for Kickstarter Projects

## Overview
This part of the project focuses on developing a clustering model to analyze Kickstarter projects based on various features. The goal is to identify patterns and segments within the data to better understand the factors contributing to the success or failure of projects.

## Data Preprocessing
* The dataset is loaded from "Kickstarter.xlsx", filtering only projects with states 'successful' or 'failed'.
* A conversion of project goals to USD is performed to standardize the financial goal of projects across different currencies.
* Missing values in the 'category' column are filled with 'Unknown', and the categories are simplified into broader groups for better analysis.
* The country of origin for each project is mapped to a region (North America, Europe, Asia-Pacific) to simplify geographical analysis.

## Feature Engineering
* Multicollinearity among numerical features is assessed using the Variance Inflation Factor (VIF), leading to the removal of highly collinear variables.
* The dataset is then prepared for clustering by encoding categorical variables and standardizing numerical variables.

## Clustering Analysis
### K-Means Clustering
* An analysis is conducted to find the optimal number of clusters for the K-Means algorithm, considering silhouette scores and the Calinski-Harabasz score.
* The optimal number of clusters is determined to be 18, with the model's silhouette score indicating a reasonable structure within the data.

### DBSCAN
* The DBSCAN algorithm is explored to identify clusters without specifying the number of clusters a priori.
* The optimal `eps` parameter is determined based on silhouette scores, with 1.5 being identified as the best value.

## Results
* Cluster labels are added to the original DataFrame, and summary statistics for each cluster are calculated. These include average USD goal, median USD pledged, most common project state, category, spotlight status, and region, as well as average launch to deadline days.
* The cluster characteristics are saved to an Excel file, "cluster_characteristics.xlsx", for further analysis and interpretation.

## Conclusion
This clustering analysis provides insights into the different segments of Kickstarter projects, potentially helping project creators understand the traits of successful and unsuccessful projects. It also aids in identifying specific areas for improvement or focus.
