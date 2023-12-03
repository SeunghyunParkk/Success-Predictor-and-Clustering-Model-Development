import pandas as pd
import numpy as np

# Task 0

df= pd.read_csv('C:/Users/Park/INSY662/OnlineNews.csv')
df = df.drop(columns=['url','timedelta'])
X = df.iloc[:,:]
y = df["popularity"]
X = X.drop(columns=['popularity'])
X_dum = pd.get_dummies(X, columns=['data_channel', 'weekdays'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X_dum)

# Task 1
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)
from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier
k = [50,100, 150, 200]
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(X_train,y_train)
    y_test_pred = model.predict(X_test)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))


# Task 2
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.05) 
model = ls.fit(X_std,y)
pd.DataFrame(list(zip(X_dum.columns,model.coef_)), columns = ['predictor','coefficient'])

ls_coefs = ls.coef_

selected_features = np.where(ls_coefs != 0)[0]

print("Selected feature indices:", selected_features)


# Task 3
from sklearn.decomposition import PCA
pca = PCA(n_components=6)

X_pca = pca.fit_transform(X_std)

# Task 4

X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(X_std, y, test_size = 0.3, random_state = 5)

# Task 5 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import timeit

knn_all = KNeighborsClassifier(n_neighbors = 200, p = 2)

knn_all_time_start = timeit.default_timer()
knn_all.fit(X_train_new, Y_train_new)
knn_all_time_end = timeit.default_timer()

knn_all_time = knn_all_time_end - knn_all_time_start

y_test_all = knn_all.predict(X_test_new)

precision_all = precision_score(Y_test_new, y_test_all)
accuracy_all = accuracy_score(Y_test_new, y_test_all)
recall_all = recall_score(Y_test_new, y_test_all)

print("All predictor KNN accuracy=",accuracy_all,", precision=",precision_all,", recall =",recall_all,"training time =",knn_all_time)

# Model 2
X_train_ls = X_train_new[:,selected_features]
X_test_ls = X_test_new[:,selected_features]

knn_ls = KNeighborsClassifier(n_neighbors = 200, p = 2)

knn_ls_time_start = timeit.default_timer()
knn_ls.fit(X_train_ls, Y_train_new)
knn_ls_time_end = timeit.default_timer()

knn_ls_time = knn_ls_time_end - knn_ls_time_start

y_test_ls = knn_ls.predict(X_test_ls)

precision_ls = precision_score(Y_test_new, y_test_ls)
accuracy_ls = accuracy_score(Y_test_new, y_test_ls)
recall_ls = recall_score(Y_test_new, y_test_ls)

print("LS predictor KNN accuracy=",accuracy_ls,", precision=",precision_ls,", recall =",recall_ls,"training time =",knn_ls_time)

# Model 3
X_train_pca = pca.fit_transform(X_train_new)
X_test_pca = pca.transform(X_test_new)

knn_pca = KNeighborsClassifier(n_neighbors = 200, p = 2)

knn_pca_time_start = timeit.default_timer()
knn_pca.fit(X_train_pca, Y_train_new)
knn_pca_time_end = timeit.default_timer()

knn_pca_time = knn_pca_time_end - knn_pca_time_start

y_test_pca = knn_pca.predict(X_test_pca)

precision_pca = precision_score(Y_test_new, y_test_pca)
accuracy_pca = accuracy_score(Y_test_new, y_test_pca)
recall_pca = recall_score(Y_test_new, y_test_pca)

print("PCA predictor KNN accuracy=",accuracy_pca,", precision=",precision_pca,", recall =",recall_pca,"training time =",knn_pca_time)

# Task 7: 
pd.DataFrame(list(zip(X_dum.columns,abs(model.coef_))), columns = ['predictor','coefficient'])

