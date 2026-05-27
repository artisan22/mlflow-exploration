import mlflow.sklearn
import pandas as pd

# Load the model from the last run
# We point to the mlruns folder where MLflow saved it
model_uri = "models:/iris-classifier/latest"

# Actually let's use the simpler direct path approach
import mlflow

# Get the latest run
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("iris-classifier")
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
latest_run_id = runs[0].info.run_id

print(f"Loading model from run: {latest_run_id}")

# Load the model
model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")

# Make a prediction on a new flower nobody has seen before
# [petal_length, petal_width, sepal_length, sepal_width]
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                           columns=["sepal length (cm)", "sepal width (cm)", 
                                    "petal length (cm)", "petal width (cm)"])

prediction = model.predict(new_flower)

species = ["setosa", "versicolor", "virginica"]
print(f"Predicted species: {species[prediction[0]]}")