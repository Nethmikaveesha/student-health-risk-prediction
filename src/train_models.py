
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


def build_logistic_regression(preprocessor, random_state=42):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="saga",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state
                )
            )
        ]
    )


def build_random_forest(preprocessor, random_state=42):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1
                )
            )
        ]
    )


def build_linear_svm(preprocessor, random_state=42):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state
                )
            )
        ]
    )


def build_mlp(random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=10,
        random_state=random_state
    )
EOF