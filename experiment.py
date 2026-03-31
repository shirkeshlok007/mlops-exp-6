import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_metric("score", model.score(X, y))
    mlflow.sklearn.log_model(model, "model")

print("Run completed!")