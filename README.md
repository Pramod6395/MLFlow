# MLFlow

#### Follow Below Steps to use MLFlow with Aws S3 as artifact Storage and Sqlite as backend store:

###### Create Virtual Environment:
```
conda create --name mlflow_tutorial
```
###### Activate Virual Environment:
```
conda activate mlflow_tutorial
```
###### Install required library given in requirements.txt
```
pip install -r requirements.txt
```
###### Create bucket name 'mlflow-95' in AWS S3

###### Create IAM user with AdministrativeAccess and download acccess and secrete keys.

###### configure credential in your machine (aws cli must be install for that purpose)

###### Start ml server with below command.
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://mlflow-95/mlflow --host 0.0.0.0 --port 5000
```
###### Now you can access MLFlow UI on below IP:
```
localhost:5000
```
###### Create file name train.py and paste below contain in it:
```
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
```
###### Run the file 'train.py' on your console.
```
python3 train.py
```
###### You can Run 'train.py' again and again just by changing values of paramter 'n_estimators' and 'max_depth' and see if we get lower mse(Mean Square Error):

###### Then can see that Experiments are logs in MLFLow and artifacts are store is S3 Bucktes.

![image](https://github.com/Pramod6395/MLFlow/assets/73251890/13424490-e7c9-49ed-a982-2bd68ae08ee1)


![image](https://github.com/Pramod6395/MLFlow/assets/73251890/c28bfe82-75a1-4598-8dce-84b336b9911d)

