
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "final_tuned_random_forest.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "final_label_encoder.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


def predict_single(input_dict):
    model, label_encoder = load_model()

    input_df = pd.DataFrame([input_dict])

    prediction_encoded = model.predict(input_df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    return prediction_label


def generate_submission(test_df, output_path):
    model, label_encoder = load_model()

    id_column = "id"
    target_column = "health_condition"

    test_ids = test_df[id_column]
    X_test = test_df.drop(columns=[id_column])

    predictions_encoded = model.predict(X_test)
    predictions_labels = label_encoder.inverse_transform(predictions_encoded)

    submission_df = pd.DataFrame({
        id_column: test_ids,
        target_column: predictions_labels
    })

    submission_df.to_csv(output_path, index=False)

    return submission_df
EOF