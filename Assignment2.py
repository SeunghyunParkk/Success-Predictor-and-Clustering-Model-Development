# -*- coding: utf-8 -*-


from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


#Task 2
data = [["Black", 1, 1], ["Blue", 0, 0], ["Blue", -1, -1]]
column_names = ["y", "x1", "x2"]

df = pd.DataFrame(data, columns=column_names)

#Task 3
X = df.iloc[:,1:3] 
y = df['y']

knn = KNeighborsClassifier(n_neighbors=2)
model = knn.fit(X, y)


#Task 4
new_obs = [[0.1, 0.1]]
print("Predicted target variable is", model.predict(new_obs)[0])

#Task 5
print("The probability that the target variable is Black is", model.predict_proba(new_obs)[0][0])
print("The probability that the target variable is Blue is", model.predict_proba(new_obs)[0][1])


# Task 6
knn = KNeighborsClassifier(n_neighbors=2, weights='distance').fit(X,y)
