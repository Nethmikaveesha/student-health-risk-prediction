
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Health Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_tuned_random_forest.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "final_label_encoder.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


model, label_encoder = load_artifacts()

st.title("🎓 Student Health Risk Prediction")
st.write(
    "This application predicts student health condition using a tuned "
    "Random Forest model trained for the Kaggle Student Health Risk task."
)

st.warning(
    "This is an educational prediction tool, not a medical diagnosis system."
)

with st.sidebar:
    st.header("Model Information")
    st.write("**Best Model:** Tuned Random Forest")
    st.write("**Metric:** Balanced Accuracy")
    st.write("**Validation Score:** 0.9453")
    st.write("**Classes:** at-risk, fit, unhealthy")

    st.divider()

    sample_type = st.selectbox(
        "Load Sample Input",
        ["Custom", "Fit-like sample", "At-risk-like sample", "Unhealthy-like sample"]
    )


samples = {
    "Custom": {
        "sleep_duration": 7.0,
        "heart_rate": 75.0,
        "bmi": 23.0,
        "calorie_expenditure": 2200.0,
        "step_count": 8000.0,
        "exercise_duration": 40.0,
        "water_intake": 2.2,
        "diet_type": "balanced",
        "stress_level": "medium",
        "sleep_quality": "average",
        "physical_activity_level": "moderate",
        "smoking_alcohol": "occasional",
        "gender": "female"
    },
    "Fit-like sample": {
        "sleep_duration": 8.0,
        "heart_rate": 72.0,
        "bmi": 22.0,
        "calorie_expenditure": 2400.0,
        "step_count": 12500.0,
        "exercise_duration": 55.0,
        "water_intake": 2.5,
        "diet_type": "balanced",
        "stress_level": "low",
        "sleep_quality": "good",
        "physical_activity_level": "active",
        "smoking_alcohol": "no",
        "gender": "female"
    },
    "At-risk-like sample": {
        "sleep_duration": 7.0,
        "heart_rate": 75.0,
        "bmi": 23.0,
        "calorie_expenditure": 2200.0,
        "step_count": 8500.0,
        "exercise_duration": 38.0,
        "water_intake": 2.1,
        "diet_type": "veg",
        "stress_level": "medium",
        "sleep_quality": "average",
        "physical_activity_level": "moderate",
        "smoking_alcohol": "occasional",
        "gender": "other"
    },
    "Unhealthy-like sample": {
        "sleep_duration": 4.8,
        "heart_rate": 84.0,
        "bmi": 28.5,
        "calorie_expenditure": 1800.0,
        "step_count": 2500.0,
        "exercise_duration": 12.0,
        "water_intake": 1.2,
        "diet_type": "non-veg",
        "stress_level": "high",
        "sleep_quality": "poor",
        "physical_activity_level": "sedentary",
        "smoking_alcohol": "yes",
        "gender": "male"
    }
}

sample = samples[sample_type]

st.subheader("Enter Student Health Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        sleep_duration = st.number_input("Sleep Duration (hours)", 3.0, 10.0, sample["sleep_duration"], 0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", 50.0, 110.0, sample["heart_rate"], 0.1)
        bmi = st.number_input("BMI", 16.0, 35.0, sample["bmi"], 0.1)
        calorie_expenditure = st.number_input("Calorie Expenditure", 1200.0, 3600.0, sample["calorie_expenditure"], 10.0)
        step_count = st.number_input("Daily Step Count", 1000.0, 15000.0, sample["step_count"], 100.0)
        exercise_duration = st.number_input("Exercise Duration (minutes)", 0.0, 100.0, sample["exercise_duration"], 0.5)
        water_intake = st.number_input("Water Intake (litres)", 0.5, 5.0, sample["water_intake"], 0.1)

    with col2:
        diet_type = st.selectbox("Diet Type", ["balanced", "veg", "non-veg"], index=["balanced", "veg", "non-veg"].index(sample["diet_type"]))
        stress_level = st.selectbox("Stress Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(sample["stress_level"]))
        sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"], index=["poor", "average", "good"].index(sample["sleep_quality"]))
        physical_activity_level = st.selectbox("Physical Activity Level", ["sedentary", "moderate", "active"], index=["sedentary", "moderate", "active"].index(sample["physical_activity_level"]))
        smoking_alcohol = st.selectbox("Smoking / Alcohol", ["no", "occasional", "yes"], index=["no", "occasional", "yes"].index(sample["smoking_alcohol"]))
        gender = st.selectbox("Gender", ["female", "male", "other"], index=["female", "male", "other"].index(sample["gender"]))

    submitted = st.form_submit_button("Predict Health Condition", use_container_width=True)


if submitted:
    input_data = pd.DataFrame({
        "sleep_duration": [sleep_duration],
        "heart_rate": [heart_rate],
        "bmi": [bmi],
        "calorie_expenditure": [calorie_expenditure],
        "step_count": [step_count],
        "exercise_duration": [exercise_duration],
        "water_intake": [water_intake],
        "diet_type": [diet_type],
        "stress_level": [stress_level],
        "sleep_quality": [sleep_quality],
        "physical_activity_level": [physical_activity_level],
        "smoking_alcohol": [smoking_alcohol],
        "gender": [gender]
    })

    prediction_encoded = model.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities.max()
    else:
        confidence = None

    st.divider()
    st.subheader("Prediction Result")

    if prediction_label == "fit":
        st.success("Predicted Health Condition: FIT")
    elif prediction_label == "at-risk":
        st.warning("Predicted Health Condition: AT-RISK")
    else:
        st.error("Predicted Health Condition: UNHEALTHY")

    if confidence is not None:
        st.metric("Model Confidence", f"{confidence * 100:.2f}%")

        probability_df = pd.DataFrame({
            "Health Condition": label_encoder.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        probability_df["Probability"] = probability_df["Probability"].map(
            lambda value: f"{value * 100:.2f}%"
        )

        st.write("### Class Probability Breakdown")
        st.dataframe(probability_df, use_container_width=True)

    st.write("### Model-Level Risk Explanation")
    st.write(
        "The Random Forest feature-importance analysis showed that the model "
        "mainly relies on sleep duration, stress level, physical activity level, "
        "BMI, exercise duration, and step count."
    )

    explanation_points = []

    if sleep_duration < 6:
        explanation_points.append("Low sleep duration may increase health-risk prediction.")
    if stress_level == "high":
        explanation_points.append("High stress level is strongly associated with unhealthy predictions.")
    if physical_activity_level == "active":
        explanation_points.append("Active physical activity is often associated with healthier predictions.")
    if step_count < 4000:
        explanation_points.append("Low step count may contribute to higher risk.")
    if exercise_duration < 20:
        explanation_points.append("Low exercise duration may contribute to higher risk.")
    if sleep_quality == "poor":
        explanation_points.append("Poor sleep quality may increase health-risk prediction.")

    if explanation_points:
        for point in explanation_points:
            st.write("- " + point)
    else:
        st.write("- No major risk signal was triggered by the simple explanation rules.")

    with st.expander("View Submitted Input Data"):
        st.dataframe(input_data, use_container_width=True)



