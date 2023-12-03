import numpy as np
import pandas as pd

#Task 1
df = pd.read_csv("ToyotaCorolla.csv")

col_name = ['Id', 'Model']
df_new = df.drop(col_name, axis=1)

Y = df_new['Price']
X = df_new.iloc[:, 1:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std_df = pd.DataFrame(X_std, columns=X.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std_df, Y, test_size=0.3, random_state=815)

#Task 2
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge = Ridge(alpha=1,random_state=815)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE =",mse)

#Task 3
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.05, random_state = 815)

pred = iforest.fit_predict(df_new)

from numpy import where
anomaly_index = where(pred == -1)[0]
anomaly_values = df_new.iloc[anomaly_index]

#Task 4
X_train_filtered = X_train.drop(X_train.index.intersection(anomaly_index))
y_train_filtered = y_train.drop(y_train.index.intersection(anomaly_index))
X_test_filtered = X_test.drop(X_test.index.intersection(anomaly_index))
y_test_filtered = y_test.drop(y_test.index.intersection(anomaly_index))

#Task 5
ridge = Ridge(alpha=1, random_state=815)
ridge.fit(X_train_filtered, y_train_filtered)

y_pred = ridge.predict(X_test_filtered)
mse = mean_squared_error(y_test_filtered, y_pred)
print("MSE =", mse)