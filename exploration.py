from train_by_features import train_by_features
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from features import categorical_features, numerical_features, binary_features, descriptions


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Richters prediction - exploration")

all_features = categorical_features + numerical_features + binary_features
for sfs in all_features:
    f1, model = train_by_features([sfs])

    with mlflow.start_run():
        # Log the hyperparameters
        params = dict()
        for i, p in enumerate([sfs]):
            params[f'cat_{i+1}'] = p
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("f1", f1)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")


        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            registered_model_name="tracking-quickstart",
        )

