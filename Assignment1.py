# -*- coding: utf-8 -*-
"""
@author: Seunghyun Park
"""

import pandas as pd

"""
===============================================================================
Task 0
===============================================================================
"""

df = pd.read_csv("ToyotaCorolla.csv")
X = df.iloc[:,3:] 
y = df['Price']
from sklearn.linear_model import LinearRegression


lm1 = LinearRegression() 
model1 = lm1.fit(X, y) 

model1.intercept_
model1.coef_

"""
===============================================================================
Task 1
===============================================================================
"""

from statsmodels.tools.tools import add_constant
X1 = add_constant(X)


vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns
  
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(X1.columns)):
    vif_data.loc[vif_data.index[i],"VIF"] = variance_inflation_factor(X1.values, i)

print(vif_data)

X = X.drop(columns=['Cylinders'])

'''
===============================================================================
Task 2
===============================================================================
'''

X.describe()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns=X.columns) # transform into a dataframe and add column names

'''
===============================================================================
Task 3
===============================================================================
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.35, random_state = 662)

'''
===============================================================================
Task 4
===============================================================================
'''

lm2 = LinearRegression()
model2 = lm2.fit(X_train,y_train)

y_test_pred = model2.predict(X_test)

from sklearn.metrics import mean_squared_error
lm2_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using linear regression = "+str(round(lm2_mse)))

'''
===============================================================================
Task 5
===============================================================================
'''

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1)
ridge_model = ridge.fit(X_train,y_train)

ridge_model.coef_

y_test_pred = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(round(ridge_mse)))

'''
===============================================================================
Task 6
===============================================================================
'''

from sklearn.linear_model import Lasso

# Run LASSO with penalty = 1
lasso = Lasso(alpha=1)
lasso_model = lasso.fit(X_train,y_train)

y_test_pred4 = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_test_pred4)
print("Test MSE using lasso with penalty of 1 = "+str(round(lasso_mse)))

'''
===============================================================================
Task 7
===============================================================================
'''

alphas = [10,100,1000,10000]
for i in alphas:

    ridge = Ridge(alpha=i)
    ridge_model = ridge.fit(X_train,y_train)

    ridge_coef = ridge_model.coef_

    y_test_pred = ridge_model.predict(X_test)

    ridge_mse = mean_squared_error(y_test, y_test_pred)
    print("Test MSE using ridge with penalty of", i ,"= "+str(round(ridge_mse)))


    lasso = Lasso(alpha=i)
    lasso_model = lasso.fit(X_train,y_train)
    lasso_coef = lasso_model.coef_

    y_test_pred4 = lasso_model.predict(X_test)


    lasso_mse = mean_squared_error(y_test, y_test_pred4)
    print("Test MSE using lasso with penalty of", i,"= "+str(round(lasso_mse)))
    
    if i == 10000:
        print("Coefficients of Ridge model (Age_08_04, KM, HP, Automatic, CC, Doors, Gears, Weight respectively):",  ridge_coef)

        print("Coefficients of Lasso model (Age_08_04, KM, HP, Automatic, CC, Doors, Gears, Weight respectively):", lasso_coef)




















