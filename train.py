import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Hyperparameters ---
N_ESTIMATORS = 10
MAX_DEPTH = 3
RANDOM_STATE = 42

# --- Data ---
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=RANDOM_STATE
)

mlflow.set_experiment("iris-classifier")
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Log test accuracy manually — autolog only covers training data
    test_accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Training complete. Test accuracy: {test_accuracy:.4f}")