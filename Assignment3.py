# -*- coding: utf-8 -*-

import pandas as pd

"""
=======================
========================================================
Task 1
===============================================================================
"""

df = pd.read_csv("Sheet4.csv")

# Checking data types
df.dtypes
df.info()

# Checking duplicates
df[df.duplicated()].shape

# Drop duplicates
df = df.drop_duplicates()
df.shape

# Irrelevant columns = Name
df = df.drop(columns=['Name'], axis=1)

# Checking missing values
print(df.isnull().sum())
df = df.dropna()

# dummy all categorical variables
df = pd.get_dummies(df,columns = ['Manuf','Type'])

"""
===============================================================================
Task 2
===============================================================================
"""
## Construct variables
X = df.drop(columns=['Rating_Binary'])
y = df['Rating_Binary']

"""
===============================================================================
Task 3
===============================================================================
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
randomforest = RandomForestClassifier(random_state=815)

hyperparameters = {'n_estimators':[50,100,150,200],'max_features':[3,4,5,6],'min_samples_leaf':[1,2,3,4]}

grid_search = GridSearchCV(estimator = randomforest, param_grid = hyperparameters,cv = 5,verbose = True)

grid_search.fit(X,y)

best_model1 = grid_search.best_estimator_
best_parameter1 = grid_search.best_params_
accuracy_score1 = grid_search.best_score_

print("The best combination of the Hyperparameters are ",best_parameter1)
print("The best model performance (accuracy score) is ", accuracy_score1)

"""
===============================================================================
Task 4
===============================================================================
"""

from sklearn.ensemble import GradientBoostingClassifier
gradientboost = GradientBoostingClassifier(random_state = 815)

hyperparameters = {'n_estimators':[50,100,150,200], 'max_features':[3,4,5,6], 'min_samples_leaf':[1,2,3,4]}

grid_search = GridSearchCV(estimator = gradientboost, param_grid = hyperparameters, cv=5,verbose=True)

grid_search.fit(X,y)

best_model2 = grid_search.best_estimator_
best_parameter2 = grid_search.best_params_
accuracy_score2 = grid_search.best_score_

print("The best combination of the Hyperparameters are ",best_parameter2)
print("The best model performance (accuracy score) is ", accuracy_score2)

































