import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

### Part 1: Data Preprocessing ###
# Load the data
df = pd.read_excel("Kickstarter.xlsx")

# Drop post-launch columns, datetime columns, and non-predictive columns
post_launch_columns = [
    "id", "name", 'pledged', 'disable_communication', 'staff_pick',
    'backers_count', 'usd_pledged', 'state_changed_at', 'state_changed_at_weekday',
    'spotlight', 'usd_pledged', 'state_changed_at_month', 'state_changed_at_day',
    'state_changed_at_yr', 'state_changed_at_hr', 'launch_to_state_change_days',
    'deadline', 'created_at', 'launched_at'
]
df.drop(columns=post_launch_columns, inplace=True)

# filter for 'successful' and 'failed' states
df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]
# map 'successful' to 1 and 'failed' to 0
df['state'] = df['state'].map({'successful': 1, 'failed': 0})

#print(df.dtypes)
# remove rows with missing values
#print(df.isnull().sum())
df.dropna(inplace=True)

# convert categorical columns to integers for machine learning analysis
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#df.head()

# Split the data into X and y
y = df['state']
X = df.drop('state', axis=1)

#### 2. Apply feature selection ####
# Standardize predictors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


## 1. LASSO
from sklearn.linear_model import Lasso
import numpy as np
# find the optimal regularization parameter using grid search
alphas = np.linspace(0.00001, 0.3, 100)
lasso = Lasso(random_state=815, max_iter= 3000)
grid_search = GridSearchCV(estimator=lasso, param_grid=dict(alpha=alphas), cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_std, y)
best_alpha = grid_search.best_estimator_.alpha
print(f"Optimal regularization parameter: {best_alpha}")

# Fit the model with the optimal regularization parameter and select features
ls = Lasso(alpha=best_alpha, random_state=815)
model = ls.fit(X_std, y)
selected_features_lasso = X.columns[(ls.coef_ != 0).ravel()].tolist()
print(selected_features_lasso)



## 2. Random Forest
# find the optimal hyperparameters using grid search
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(random_state=815)
hyperparameters = {'n_estimators':[50,100,150,200],'max_features':[3,4,5,6],'min_samples_leaf':[1,2,3,4]}
grid_search = GridSearchCV(estimator = randomforest, param_grid = hyperparameters,cv = 5,verbose = True)
grid_search.fit(X,y)
best_parameter1 = grid_search.best_params_
print("The best combination of the Hyperparameters are ",best_parameter1)

# using optimal hyperparameters, find features using Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=6, max_features=3, min_samples_leaf=50, random_state=815)
randomforest.fit(X, y)
selected_features_randomforest = X.columns[randomforest.feature_importances_ > 0.05].tolist()
print(selected_features_randomforest)



### 3. Hgperparameter tuning ###
# Models to tune
models = {'RandomForest': RandomForestClassifier(random_state=815),'GradientBoosting': GradientBoostingClassifier(random_state=815)}
# Feature sets
feature_sets = {'Lasso': selected_features_lasso,'RandomForest': selected_features_randomforest}
# Hyperparameters for tuning
hyperparameters = {'RandomForest': {'n_estimators':[50,100,150,200],'max_features':[3,4,5,6],'min_samples_leaf':[1,2,3,4]},
'GradientBoosting': {'n_estimators':[50,100,150,200], 'max_features':[3,4,5,6], 'min_samples_leaf':[1,2,3,4]}}

# Tuning each model and storing the best models
best_models = {}
for model_name, model in models.items():
    best_models[model_name] = {}
    for feature_name, features in feature_sets.items():
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters[model_name], cv=5, verbose=True)
        grid_search.fit(X[features], y)
        best_model = grid_search.best_estimator_
        best_models[model_name][feature_name] = best_model
        print(f"Tuned {model_name} using {feature_name} features")

print(best_models)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=815)

# Training and Evaluating Models
for model_name, model_dict in best_models.items():
    print(f"Evaluating {model_name} with different feature selection methods:")
    for feature_name, tuned_model in model_dict.items():
        tuned_model.fit(X_train[feature_sets[feature_name]], y_train)
        accuracy = accuracy_score(y_test, tuned_model.predict(X_test[feature_sets[feature_name]]))
        print(f" - Accuracy with {feature_name} selected features: {accuracy}")
    print("\n")

# Final Model: Gradient Boosting with Random Forest features
models = {'GradientBoosting': GradientBoostingClassifier(random_state=815)}

# Only include selected_features_randomforest for feature_sets
feature_sets = {'RandomForest': selected_features_randomforest}

# Hyperparameters for tuning
hyperparameters = {
    'GradientBoosting': {
        'n_estimators': [50, 100, 150, 200],
        'max_features': [3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4]
    }
}

# Tuning each model and storing the best models
best_models = {}
for model_name, model in models.items():
    best_models[model_name] = {}
    for feature_name, features in feature_sets.items():
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters[model_name], cv=5, verbose=True)
        grid_search.fit(X[features], y)
        best_model = grid_search.best_estimator_
        best_models[model_name][feature_name] = best_model
        print(f"Tuned {model_name} using {feature_name} features")

print(best_models)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=815)

# Training and Evaluating Models
for model_name, model_dict in best_models.items():
    print(f"Evaluating {model_name} with different feature selection methods:")
    for feature_name, tuned_model in model_dict.items():
        tuned_model.fit(X_train[feature_sets[feature_name]], y_train)
        accuracy = accuracy_score(y_test, tuned_model.predict(X_test[feature_sets[feature_name]]))
        print(f" - Accuracy with {feature_name} selected features: {accuracy}")
    print("\n")

# Final Model: Gradient Boosting with Random Forest features
y = df['state']
X = df[['goal', 'category', 'name_len', 'name_len_clean', 'create_to_launch_days', 'launch_to_deadline_days']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=815)

final_model = GradientBoostingClassifier(max_features=4, min_samples_leaf=2, n_estimators=200, random_state=815)
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_test_pred)
print("Final model's accuracy is", final_accuracy)
print("\n")

######################For Out-Of-Sample Validation########################################################################################
import pandas as pd
from sklearn.metrics import accuracy_score


grading_df = pd.read_excel("Kickstarter-Grading-Sample.xlsx")

# Drop post-launch columns, datetime columns, and non-predictive columns
post_launch_columns = [
    "id", "name", 'pledged', 'disable_communication', 'staff_pick',
    'backers_count', 'usd_pledged', 'state_changed_at', 'state_changed_at_weekday',
    'spotlight', 'usd_pledged', 'state_changed_at_month', 'state_changed_at_day',
    'state_changed_at_yr', 'state_changed_at_hr', 'launch_to_state_change_days',
    'deadline', 'created_at', 'launched_at'
]
grading_df.drop(columns=post_launch_columns, inplace=True)

# filter for 'successful' and 'failed' states
grading_df = grading_df[(grading_df['state'] == 'successful') | (grading_df['state'] == 'failed')]
# map 'successful' to 1 and 'failed' to 0
grading_df['state'] = grading_df['state'].map({'successful': 1, 'failed': 0})

# remove rows with missing values
grading_df.dropna(inplace=True)

# convert categorical columns to integers for machine learning analysis
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in grading_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    grading_df[col] = le.fit_transform(grading_df[col])
    label_encoders[col] = le

# Split the data into X and y
y_grading = grading_df['state']
X_grading = grading_df[['goal', 'category', 'name_len', 'name_len_clean', 'create_to_launch_days', 'launch_to_deadline_days']]

y_grading_pred = final_model.predict(X_grading)
accuracy_grading = accuracy_score(y_grading, y_grading_pred)
print("For Grading: Accuracy is ", accuracy_grading)


########################################################################################################################
######## Developiong a Clustering Model#################################################################################
import pandas as pd
import numpy as np

# Load the data once
df = pd.read_excel("Kickstarter.xlsx")

# Filter for 'successful' and 'failed' states
df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]

# Convert goal to USD using vectorized operation
for index, row in df.iterrows():
    if row['currency'] == 'USD':
        df.at[index, 'usd_goal'] = row['goal']
    else:
        df.at[index, 'usd_goal'] = row['goal'] * row['static_usd_rate']

# Check the number of values in category
print(df['category'].value_counts())
# Fill missing 'category' values as 'Unknown'
df['category'].fillna('Unknown', inplace = True)

# Simplify the category column (Technology, Arts & Entertainment, Other)
category_simplification = {
    'Hardware': 'Technology',
    'Web': 'Technology',
    'Software': 'Technology',
    'Unknown': 'Other',
    'Gadgets': 'Technology',
    'Plays': 'Arts & Entertainment',
    'Apps': 'Technology',
    'Musical': 'Arts & Entertainment',
    'Wearables': 'Technology',
    'Festivals': 'Arts & Entertainment',
    'Robots': 'Technology',
    'Sound': 'Technology',
    'Flight': 'Technology',
    'Experimental': 'Arts & Entertainment',
    'Immersive': 'Arts & Entertainment',
    'Makerspaces': 'Arts & Entertainment',
    'Spaces': 'Arts & Entertainment',
    'Places': 'Arts & Entertainment',
    'Shorts': 'Arts & Entertainment',
    'Webseries': 'Arts & Entertainment',
    'Academic': 'Other',
    'Thrillers': 'Other',
    'Blues': 'Other'
}

# Apply the mapping to the category column
df['simplified_category'] = df['category'].map(category_simplification)


# Check the number of values in country
print(df['country'].value_counts())
# simplify the country column (North America, Europe, Asia-Pacific)
country_to_region = {
    'US': 'NA',
    'GB': 'EU',
    'CA': 'NA',
    'AU': 'APAC',
    'NL': 'EU',
    'DE': 'EU',
    'FR': 'EU',
    'IT': 'EU',
    'DK': 'EU',
    'NZ': 'APAC',
    'ES': 'EU',
    'SE': 'EU',
    'IE': 'EU',
    'NO': 'EU',
    'CH': 'EU',
    'BE': 'EU',
    'AT': 'EU',
    'LU': 'EU'
}

# Apply the mapping to the country column
df['region'] = df['country'].map(country_to_region)

# Check multicollinearity of interested columns using VIF
X_num= df[['usd_goal', 'usd_pledged', 'backers_count', 'launch_to_deadline_days']]
from statsmodels.tools.tools import add_constant
X1 = add_constant(X_num)
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns

# Calculating VIF for each feature
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(X1.columns)):
    vif_data.loc[vif_data.index[i], "VIF"] = variance_inflation_factor(X1.values, i)
print(vif_data)
# Since usd_pledged and backers_count have high VIF, remove backers_count

# Interested columns, only choosing the columns that are useful for business perspective
X = df[['state','usd_goal', 'usd_pledged', 'simplified_category', 'region', 'spotlight','launch_to_deadline_days']]
# Dummify  categorical columns
X = pd.get_dummies(X, columns = ['state','simplified_category', 'region','spotlight'])

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


# Silhouette analysis for K-Means to find the optimal number of clusters
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import f
# Finding optimal K
for i in range (5,30):
    df1=i-1
    df2=22-i
    kmeans = KMeans(n_clusters=i, random_state = 815)
    model = kmeans.fit(X_std)
    labels = model.labels_
    score = calinski_harabasz_score(X_std, labels)
    print(i,'F-score:',score)
    print(i,'p-value:',1-f.cdf(score, df1, df2))
    print(i, 'silhouette score:', silhouette_score(X_std, labels))


optimal_k = 18
print(f"Optimal number of clusters for K-Means: {optimal_k}")

# Assuming X_std is your standardized data and optimal_k is the optimal number of clusters you've determined
kmeans = KMeans(n_clusters=optimal_k, random_state=815)
kmeans_labels = kmeans.fit_predict(X_std)  # Fit the model to your data

# Calculate silhouette score
kmeans_silhouette = silhouette_score(X_std, kmeans_labels)
print("k-means clustering method with k = 18 silhouette score:", kmeans_silhouette)


# find the optimal eps value for DBSCAN method using silhouette score
eps_values = np.arange(0.1, 2.0, 0.1)
for i in eps_values:
    dbscan = DBSCAN(eps=i, min_samples=30)
    model = dbscan.fit(X_std)
    labels = model.labels_
    print('eps:',round(i,2),'silhouette score:',silhouette_score(X_std, labels))

optimal_eps = 1.5
print("Optimal eps for DBSCAN:", optimal_eps)

# Apply DBSCAN with the optimal eps and calculate silhouette score
dbscan = DBSCAN(eps=optimal_eps, min_samples=30)
dbscan_labels = dbscan.fit_predict(X_std)
dbscan_silhouette = silhouette_score(X_std,dbscan_labels)
print("DBSCAN clustering method with eps=1.5 silhouette score:", dbscan_silhouette)

# Add the cluster labels to the DataFrame
df['dbscan_cluster'] = dbscan_labels
cluster_groups = df.groupby('dbscan_cluster')

cluster_characteristics = pd.DataFrame()

# Loop through each cluster and calculate summary statistics
for cluster, group in cluster_groups:
    cluster_summary = pd.DataFrame({
        'Cluster': [cluster],
        'Average USD Goal': [group['usd_goal'].mean()],
        'Median USD Pledged': [group['usd_pledged'].median()],
        'Most Common State': [group['state'].mode()[0] if not group['state'].mode().empty else np.nan],
        'Most Common Category': [group['simplified_category'].mode()[0] if not group['simplified_category'].mode().empty else np.nan],
        'Most Common Spotlight': [group['spotlight'].mode()[0] if not group['spotlight'].mode().empty else np.nan],
        'Most Common Region': [group['region'].mode()[0] if not group['region'].mode().empty else np.nan],
        'Average Launch to Deadline Days': [group['launch_to_deadline_days'].mean()]
    })
    cluster_characteristics = pd.concat([cluster_characteristics, cluster_summary], ignore_index=True)

# Set the cluster number as the index
cluster_characteristics.set_index('Cluster', inplace=True)
pd.set_option('display.max_columns', None)
# Display the DataFrame
print(cluster_characteristics)

# Save the DataFrame to an Excel file
cluster_characteristics.to_excel("cluster_characteristics.xlsx")
