import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



# Set the MLflow tracking URI to the desired location
mlflow.set_tracking_uri("http://localhost:5000")

# Start an MLflow experiment
mlflow.set_experiment("Housing_price")

# Loading the California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting parameters
params = {
    "n_estimators": 400,
    "learning_rate": 0.2,
    "max_depth": 5,
    "random_state": 42,
}

# Using MLFlow to keep track of our runs
with mlflow.start_run():

    # Logging model parameters
    for param, value in params.items():
        mlflow.log_param(param, value)

    # Creating and training a Gradient Boosting model
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    # Predicting on the testing set
    y_pred = model.predict(X_test)

    # Calculating and logging the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # Logging the model
    mlflow.sklearn.log_model(model, "model")

    # Print run ID
    run_id = mlflow.active_run().info.run_id
    print("MLflow run ID:", run_id)
