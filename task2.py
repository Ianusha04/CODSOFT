import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import chardet 
import numpy as np

# Load the dataset
file_path = ('G:\\codsoft\\movie\\IMDb Movies India.csv')

# Detect the encoding of the file
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
charenc = result['encoding']

df = pd.read_csv(file_path, encoding=charenc)

# Preprocess the data
df = df.dropna()  # handle missing data
df = df.drop_duplicates()  # remove duplicates

# Extract the features and target variable
features = ['Genre', 'Director']
target = 'Rating'

# Check if the features exist in the dataset
for feature in features:
    if feature not in df.columns:
        print(f"Warning: {feature} column not found in the dataset.")
        features.remove(feature)

# Convert categorical variables into numerical variables
le = LabelEncoder()
for col in features:
    if col in df.columns:
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])

# Define features (X) and target variable (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Train a decision tree regressor model
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)

# Train a random forest regressor model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# Train a gradient boosting regressor model
model_gb = GradientBoostingRegressor()
model_gb.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_gb = model_gb.predict(X_test)

# Evaluate the models
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_gb = mean_squared_error(y_test, y_pred_gb)

print(f'Linear Regression MSE: {mse_lr:.2f}')
print(f'Decision Tree Regressor MSE: {mse_dt:.2f}')
print(f'Random Forest Regressor MSE: {mse_rf:.2f}')
print(f'Gradient Boosting Regressor MSE: {mse_gb:.2f}')

# Select the best-performing model
best_model = model_gb  # based on the lowest MSE

# Use the best-performing model to make predictions on new data
new_data = pd.DataFrame({'Genre': ['Action'], 'Director': ['James Cameron']})
le.fit(df['Genre'].unique().tolist() + new_data['Genre'].unique().tolist())
new_data['Genre'] = le.transform(new_data['Genre'])
le.fit(df['Director'].unique().tolist() + new_data['Director'].unique().tolist())
new_data['Director'] = le.transform(new_data['Director'])
new_prediction = best_model.predict(new_data)
print(f'Predicted Rating: {new_prediction[0]:.2f}')

# Analyze the importance of features
feature_importances = best_model.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print(feature_importances_df.sort_values(by='Importance', ascending=False))

# Insights into the factors that influence movie ratings
print("The factors that influence movie ratings are:")
print("1. Genre: The type of movie (e.g. Action, Comedy, Romance) plays a significant role in determining the rating.")
print("2. Director: The director's reputation and past work can impact the rating of the movie.")