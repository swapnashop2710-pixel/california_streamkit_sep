import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("TRAINING SCRIPT STARTED")

df = fetch_california_housing(as_frame=True).frame

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1234
)

model = LinearRegression()
model.fit(X_train, y_train)

obj = {
    'model': model,
    'columns': X.columns
}

joblib.dump(obj, 'california.joblib')

print("Model saved successfully!")