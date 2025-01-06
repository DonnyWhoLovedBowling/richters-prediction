import mlflow
from train_by_features import train_by_features, train_nn_by_features, create_submission
from features import categorical_features, numerical_features, binary_features

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment

# Retrieve all runs from a specific experiment
features_sorted = (mlflow.search_runs(experiment_names=['Richters prediction - exploration']).
                    sort_values(by=['metrics.f1'],ascending=False))
n_features = 11
top_features = features_sorted.head(n_features)['params.cat_1']
other_features =  features_sorted.tail(len(features_sorted)-n_features)['params.cat_1']

mlflow.set_experiment("Richters prediction - final model - nn")

f1, model = train_nn_by_features(top_features)
create_submission(model, top_features)
exit()
# Export the DataFrame to a CSV file
for of in other_features:
    training_features = list(top_features) + [of]
    f1, model = train_nn_by_features(training_features)

    with mlflow.start_run():
        # Log the hyperparameters
        params = dict()
        for i, p in enumerate(training_features):
            params[f'cat_{i+1}'] = p
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("f1", f1)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")


    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="final_model_top_5_with_1_extra_neural_network",
        registered_model_name="final_model_top_5_with_1_extra_neural_network",
    )

