from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Health Risk Prediction",
    page_icon="🎓",
    layout="wide"
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_tuned_random_forest.pkl"
)

ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_label_encoder.pkl"
)


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    return model, label_encoder


try:
    model, label_encoder = load_artifacts()

except FileNotFoundError:
    st.error(
        "Model files were not found. "
        "Run the final model submission notebook first."
    )
    st.stop()

except Exception as error:
    st.error(
        f"Unable to load model artifacts: {error}"
    )
    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🎓 Student Health Risk Prediction")

st.write(
    "This application predicts a student's health condition "
    "using lifestyle and health-related indicators."
)

st.info(
    "The prediction model is a tuned Random Forest classifier "
    "selected using Balanced Accuracy."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Model Information")

    st.write("**Model:** Tuned Random Forest")
    st.write("**Task:** Multiclass Classification")
    st.write("**Primary Metric:** Balanced Accuracy")
    st.write("**Validation Balanced Accuracy:** 0.9453")

    st.divider()

    st.caption(
        "Educational computational intelligence project."
    )


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Enter Student Health Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        sleep_duration = st.number_input(
            "Sleep Duration (hours)",
            min_value=3.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=50.0,
            max_value=110.0,
            value=75.0,
            step=0.1
        )

        bmi = st.number_input(
            "BMI",
            min_value=16.0,
            max_value=35.0,
            value=23.0,
            step=0.1
        )

        calorie_expenditure = st.number_input(
            "Calorie Expenditure",
            min_value=1200.0,
            max_value=3600.0,
            value=2200.0,
            step=10.0
        )

        step_count = st.number_input(
            "Daily Step Count",
            min_value=1000.0,
            max_value=15000.0,
            value=8000.0,
            step=100.0
        )

        exercise_duration = st.number_input(
            "Exercise Duration (minutes)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.5
        )

        water_intake = st.number_input(
            "Water Intake (litres)",
            min_value=0.5,
            max_value=5.0,
            value=2.2,
            step=0.1
        )

    with col2:
        diet_type = st.selectbox(
            "Diet Type",
            options=[
                "balanced",
                "veg",
                "non-veg"
            ]
        )

        stress_level = st.selectbox(
            "Stress Level",
            options=[
                "low",
                "medium",
                "high"
            ]
        )

        sleep_quality = st.selectbox(
            "Sleep Quality",
            options=[
                "poor",
                "average",
                "good"
            ]
        )

        physical_activity_level = st.selectbox(
            "Physical Activity Level",
            options=[
                "sedentary",
                "moderate",
                "active"
            ]
        )

        smoking_alcohol = st.selectbox(
            "Smoking / Alcohol",
            options=[
                "no",
                "occasional",
                "yes"
            ]
        )

        gender = st.selectbox(
            "Gender",
            options=[
                "female",
                "male",
                "other"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Health Condition",
        use_container_width=True
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:

    input_data = pd.DataFrame(
        {
            "sleep_duration": [sleep_duration],
            "heart_rate": [heart_rate],
            "bmi": [bmi],
            "calorie_expenditure": [
                calorie_expenditure
            ],
            "step_count": [step_count],
            "exercise_duration": [
                exercise_duration
            ],
            "water_intake": [water_intake],
            "diet_type": [diet_type],
            "stress_level": [stress_level],
            "sleep_quality": [sleep_quality],
            "physical_activity_level": [
                physical_activity_level
            ],
            "smoking_alcohol": [
                smoking_alcohol
            ],
            "gender": [gender]
        }
    )

    prediction_encoded = model.predict(
        input_data
    )

    prediction_label = (
        label_encoder.inverse_transform(
            prediction_encoded
        )[0]
    )

    st.divider()

    st.subheader("Prediction Result")

    if prediction_label == "fit":
        st.success(
            "Predicted Health Condition: FIT"
        )

    elif prediction_label == "at-risk":
        st.warning(
            "Predicted Health Condition: AT-RISK"
        )

    else:
        st.error(
            "Predicted Health Condition: UNHEALTHY"
        )

    with st.expander(
        "View Submitted Input Data"
    ):
        st.dataframe(
            input_data,
            use_container_width=True
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Student Health Risk Prediction | "
    "Computational Intelligence Project"
)