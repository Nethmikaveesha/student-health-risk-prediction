
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_classifier(model_name, y_true, y_pred, class_names=None):
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    }

    print("=" * 70)
    print(model_name)
    print("=" * 70)

    for key, value in results.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    return results


def save_results(results, output_path):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
EOF