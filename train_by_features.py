from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


def train_gbc_by_features(selected_features: list[str]=None) -> int:
    train_values_subset = train_values[selected_features]
    train_values_subset = pd.get_dummies(train_values_subset)

    pipe = make_pipeline(StandardScaler(),
                         GradientBoostingClassifier(random_state=2018, ))

    in_sample_preds = cross_val_predict(pipe,
                                 train_values_subset,
                                 train_labels,
                                 cv=5,
                                 method="predict")

    return f1_score(train_labels, in_sample_preds, average='micro')

def train_bag_by_features(selected_features: list[str]=None) -> tuple:
    train_values_subset = train_values[selected_features]
    train_values_subset = pd.get_dummies(train_values_subset)
    df_mv = train_values_subset[train_values_subset.isna().any(axis=1)]
    print(f"missing values: {len(df_mv)} out of {len(train_values_subset)}")
    print(df_mv.head())
    pipe = make_pipeline(StandardScaler(),
                         BaggingClassifier(random_state=2018, n_jobs=-1))

    param_grid = {
        'baggingclassifier__max_features': [0.5, 0.75, 1.0],
        'baggingclassifier__max_samples': [0.65, 1.0],
        'baggingclassifier__n_estimators': [80, 100, 120],
        'baggingclassifier__bootstrap': [True],
        'baggingclassifier__bootstrap_features': [True, False],

    }
    f1_scorer = make_scorer(f1_score, average='micro')

    gs = GridSearchCV(pipe, param_grid, cv=2, verbose=3, scoring=f1_scorer)
    gs.fit(train_values_subset, train_labels.values.ravel())
    in_sample_preds = gs.predict(train_values_subset)
    return f1_score(train_labels, in_sample_preds, average='micro'), gs

def train_sgd_by_features(selected_features: list[str]=None) -> tuple:
    train_values_subset = train_values[selected_features]
    train_values_subset = pd.get_dummies(train_values_subset)
    df_mv = train_values_subset[train_values_subset.isna().any(axis=1)]

    pipe = make_pipeline(StandardScaler(),
                         RBFSampler(random_state=2018, gamma=1.0),
                         SGDClassifier(max_iter=5))

    param_grid = {
        'rbfsampler__gamma': [0.25, 0.5, 1.0],
        'sgdclassifier__max_iter': [5, 10],
    }
    f1_scorer = make_scorer(f1_score, average='micro')

    gs = GridSearchCV(pipe, param_grid, cv=2, verbose=3, scoring=f1_scorer)
    gs.fit(train_values_subset, train_labels.values.ravel())
    in_sample_preds = gs.predict(train_values_subset)
    return f1_score(train_labels, in_sample_preds, average='micro'), gs


def train_by_features(selected_features: list[str]=None) -> int:
    train_values_subset = train_values[selected_features]

    train_values_subset = pd.get_dummies(train_values_subset)

    pipe = make_pipeline(StandardScaler(),
                         RandomForestClassifier(random_state=2018, min_samples_leaf=3))

    # param_grid = {'randomforestclassifier__n_estimators': [100],
    #               'randomforestclassifier__min_samples_leaf': [3]}
    # gs = GridSearchCV(pipe, param_grid, cv=2, verbose=3)
    # gs.fit(train_values_subset, train_labels.values.ravel())
    model = pipe.fit(train_values_subset, train_labels.values.ravel())

    in_sample_preds = model.predict(train_values_subset)
    return f1_score(train_labels, in_sample_preds, average='micro'), model


def train_nn_by_features(selected_features: list[str]=None) -> int:
    train_values_subset = train_values[selected_features]
    print(train_values_subset.head())

    train_values_subset = pd.get_dummies(train_values_subset)

    clf = MLPClassifier(random_state=1, max_iter=2000)
    pipe = Pipeline(steps=[('scale', PowerTransformer()),
                           ('mlpc', clf)])
    param_grid = {
        # 'cv__max_df': (0.9, 0.95, 0.99),
        # 'cv__min_df': (0.01, 0.05, 0.1),
        'mlpc__alpha': 10.0 ** -np.arange(2, 3),
        'mlpc__hidden_layer_sizes': ((32, 64), (64, 64), (128, 64))
    }

    gs = GridSearchCV(pipe, param_grid, cv=2,verbose=50)
    labels = train_labels.values.ravel()
    gs.fit(train_values_subset, labels)
    print(gs.best_params_)
    in_sample_preds = gs.predict(train_values_subset)
    return f1_score(train_labels, in_sample_preds, average='micro'), gs

def create_submission(model, selected_features):
    test_values_subset = test_values[selected_features]
    test_values_subset = pd.get_dummies(test_values_subset)
    test_labels: pd.DataFrame = model.predict(test_values_subset)
    test_values_subset['damage_grade'] = test_labels
    test_values_subset = test_values_subset[['damage_grade']]
    test_values_subset.to_csv(DATA_DIR / 'submission.csv')

DATA_DIR = Path('data')
train_values = pd.read_csv(DATA_DIR / 'train_values.csv', index_col='building_id')
train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv', index_col='building_id')
test_values = pd.read_csv(DATA_DIR / 'test_values.csv', index_col='building_id')
