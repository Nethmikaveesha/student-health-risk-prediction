# Student Health Risk Prediction

An undergraduate Computational Intelligence project for predicting student health conditions using machine learning and neural network approaches.

The project is based on the Kaggle competition **Playground Series - Season 6, Episode 7: Predicting Student Health Risk** and compares four classification model families:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine
4. MLP Neural Network

The final model was selected using **Balanced Accuracy**, which is appropriate for the strongly imbalanced multiclass target distribution.

## Competition Information

- **Competition:** Predicting Student Health Risk
- **Series:** Playground Series - Season 6, Episode 7
- **Platform:** Kaggle
- **Competition URL:** https://www.kaggle.com/competitions/playground-series-s6e7
- **Target Column:** `health_condition`
- **Target Classes:** `at-risk`, `fit`, `unhealthy`
- **Official Evaluation Metric:** Balanced Accuracy

## Project Objective

The objective of this project is to develop and compare Computational Intelligence and machine learning approaches for multiclass student health-risk prediction.

The project investigates:

- class imbalance in student health data;
- missing-value handling;
- numerical and categorical feature preprocessing;
- linear classification;
- ensemble learning;
- margin-based classification;
- neural network learning;
- hyperparameter tuning;
- model evaluation using imbalance-aware metrics;
- final Kaggle submission generation;
- interactive prediction through a Streamlit application.

## Dataset Overview

The competition dataset contains health, behavioural, and lifestyle-related indicators.

### Dataset Sizes

| Dataset | Shape |
|---|---:|
| Training Data | 690,088 × 15 |
| Test Data | 295,753 × 14 |
| Sample Submission | 295,753 × 2 |

### Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| at-risk | 592,561 | 85.87% |
| unhealthy | 57,724 | 8.36% |
| fit | 39,803 | 5.77% |

The majority-to-minority ratio is approximately **14.89:1**, showing substantial class imbalance.

## Input Features

### Numerical Features

- `sleep_duration`
- `heart_rate`
- `bmi`
- `calorie_expenditure`
- `step_count`
- `exercise_duration`
- `water_intake`

### Categorical Features

- `diet_type`
- `stress_level`
- `sleep_quality`
- `physical_activity_level`
- `smoking_alcohol`
- `gender`

The `id` column is retained for submission generation but excluded from model training.

## Exploratory Data Analysis

The EDA stage investigates:

- dataset dimensions and data types;
- target-class distribution;
- missing-value patterns;
- numerical descriptive statistics;
- skewness;
- potential outliers using the IQR method;
- categorical feature distributions;
- categorical feature relationships with the target;
- class-wise numerical summaries;
- numerical feature correlations;
- missingness patterns across target classes.

Important observations include:

- substantial target-class imbalance;
- approximately 4.34% overall missing values;
- `stress_level` and `sleep_duration` contain the largest missing percentages;
- no duplicate training rows were detected;
- `sleep_duration`, `stress_level`, and `physical_activity_level` show important relationships with the target;
- numerical feature correlations are generally low to moderate.

## Data Preprocessing

The preprocessing workflow includes:

### Numerical Features

- median imputation for missing values;
- standardisation for scale-sensitive models.

### Categorical Features

- constant-value imputation using `Missing`;
- one-hot encoding;
- unknown-category handling using `handle_unknown="ignore"`.

### Additional Preparation

- `id` removed from model predictors;
- target labels encoded using `LabelEncoder`;
- stratified train-validation split;
- random state fixed at `42`;
- class imbalance handled using class weighting or balanced training strategies where appropriate.

## Models Developed

### 1. Logistic Regression

Used as an interpretable linear baseline.

Variants:

- Baseline Logistic Regression
- Class-Weighted Logistic Regression
- Tuned Logistic Regression

Best variant:

- **Class-Weighted Logistic Regression**

### 2. Random Forest

Used as a non-linear ensemble classifier.

Variants:

- Baseline Random Forest
- Class-Weighted Random Forest
- Tuned Random Forest

Best tuned parameters:

```text
n_estimators = 200
max_depth = 15
min_samples_split = 2
min_samples_leaf = 4
max_features = sqrt
class_weight = balanced
```

Best variant:

- **Tuned Random Forest**

### 3. Support Vector Machine

A Linear SVM implementation was used for computational feasibility on the large dataset.

Variants:

- Baseline Linear SVM
- Class-Weighted Linear SVM
- Tuned Linear SVM

Best variant:

- **Class-Weighted Linear SVM**

### 4. MLP Neural Network

Used for non-linear representation learning.

Variants:

- Baseline MLP Neural Network
- Balanced-Training MLP Neural Network
- Tuned MLP Neural Network

Best tuned architecture:

```text
Hidden Layers = (64, 32)
Alpha = 0.0001
```

Best variant:

- **Tuned MLP Neural Network**

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Class-wise Precision
- Class-wise Recall
- Class-wise F1-score
- Confusion Matrix
- Classification Report

## Why Balanced Accuracy?

The dataset is highly imbalanced:

- `at-risk`: 85.87%
- `unhealthy`: 8.36%
- `fit`: 5.77%

Ordinary accuracy can be dominated by the majority class. Balanced Accuracy gives equal importance to recall performance across all target classes.

Therefore, **Balanced Accuracy was used as the primary final model-selection metric** and is also aligned with the Kaggle competition evaluation objective.

## Best Model Per Family

| Rank | Model Family | Best Variant | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| 1 | Random Forest | Tuned Random Forest | 0.9481 | 0.9453 | 0.8798 |
| 2 | MLP Neural Network | Tuned MLP Neural Network | 0.9347 | 0.9453 | 0.8568 |
| 3 | Logistic Regression | Class-Weighted Logistic Regression | 0.8334 | 0.8989 | 0.7064 |
| 4 | SVM | Class-Weighted Linear SVM | 0.8928 | 0.8814 | 0.7731 |

## Final Model Selection

The final selected model is:

**Tuned Random Forest**

Validation performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.948108 |
| Balanced Accuracy | 0.945314 |
| Macro Precision | 0.829592 |
| Macro Recall | 0.945314 |
| Macro F1 | 0.879829 |

The Tuned Random Forest achieved the highest Balanced Accuracy among the compared models.

The Tuned MLP Neural Network achieved a very close Balanced Accuracy of `0.945256`, with an absolute difference of approximately `0.000058`.

Random Forest was selected because it achieved:

- the highest Balanced Accuracy;
- stronger Macro F1 than the tuned MLP;
- high recall across all three classes;
- useful feature-importance information;
- direct compatibility with probability-based predictions.

## Important Random Forest Features

Feature-importance analysis identified important predictors including:

- sleep duration;
- stress level;
- physical activity level;
- BMI;
- exercise duration;
- step count;
- sleep quality;
- calorie expenditure.

The strongest transformed feature was `sleep_duration`, followed by multiple stress-level and physical-activity indicators.

Feature importance represents model-level predictive importance and should not be interpreted as proof of medical causation.

## Final Kaggle Submission Workflow

The final submission notebook:

```text
notebooks/09_final_model_submission.ipynb
```

performs the following steps:

1. loads training, test, and sample submission datasets;
2. prepares the 13 model predictors;
3. encodes target labels;
4. builds the final preprocessing pipeline;
5. trains the Tuned Random Forest on the full training dataset;
6. generates predictions for 295,753 competition test rows;
7. converts encoded predictions back to original labels;
8. validates the submission format;
9. saves the final Kaggle-ready CSV file.

Validation checks:

```text
Columns Match: True
Row Count Match: True
IDs Match: True
Labels Valid: True
All submission validation checks passed.
```

Final submission path:

```text
outputs/submissions/submission.csv
```

## Streamlit Prediction Application

An interactive Streamlit application is included for demonstration.

Features include:

- manual student health input;
- sample input presets;
- health-condition prediction;
- class probability breakdown;
- prediction confidence display;
- model-level risk explanation;
- submitted input preview;
- educational-use disclaimer.

Run the application using:

```bash
python3 -m streamlit run app/app.py
```

Then open the local Streamlit address displayed in the terminal, normally:

```text
http://localhost:8501
```

## Project Architecture

```text
Raw Kaggle Data
       |
       v
Competition Understanding
       |
       v
Exploratory Data Analysis
       |
       v
Data Preprocessing
       |
       v
+-----------------------------+
| Logistic Regression         |
| Random Forest               |
| Support Vector Machine      |
| MLP Neural Network          |
+-----------------------------+
       |
       v
Model Evaluation
       |
       v
Balanced Accuracy Comparison
       |
       v
Best Model Selection
       |
       v
Tuned Random Forest
       |
       +----------------------+
       |                      |
       v                      v
Kaggle Submission       Streamlit App
```

## Project Structure

```text
student-health-risk-prediction/
|
├── app/
│   └── app.py
|
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
|
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── best_svm.pkl
│   ├── best_mlp.pkl
│   ├── best_model.pkl
│   ├── final_tuned_random_forest.pkl
│   └── final_label_encoder.pkl
|
├── notebooks/
│   ├── 01_competition_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_svm.ipynb
│   ├── 07_mlp_neural_network.ipynb
│   ├── 08_model_comparison.ipynb
│   └── 09_final_model_submission.ipynb
|
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── submissions/
│   │   └── submission.csv
│   ├── all_model_comparison.csv
│   └── best_model_per_family.csv
|
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
|
├── report/
├── requirements.txt
├── README.md
└── .gitignore
```

## Notebook Execution Order

Run notebooks in this order:

```text
01_competition_understanding.ipynb
        |
        v
02_eda.ipynb
        |
        v
03_data_preprocessing.ipynb
        |
        v
04_logistic_regression.ipynb
        |
        v
05_random_forest.ipynb
        |
        v
06_svm.ipynb
        |
        v
07_mlp_neural_network.ipynb
        |
        v
08_model_comparison.ipynb
        |
        v
09_final_model_submission.ipynb
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd student-health-risk-prediction
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Reproducible Environment

The project was developed using:

```text
Python 3.14.0
NumPy 2.4.6
Pandas 3.0.3
Scikit-learn 1.8.0
Joblib 1.5.3
Streamlit 1.59.0
Matplotlib 3.10.9
IPykernel 7.2.0
Jupyter Client 8.8.0
Jupyter Core 5.9.1
```

Exact package versions are pinned in:

```text
requirements.txt
```

## Required Model Files

The Streamlit application requires:

```text
models/final_tuned_random_forest.pkl
models/final_label_encoder.pkl
```

Large trained model files may be excluded from GitHub because of repository file-size limitations.

If model files are unavailable, run:

```text
notebooks/09_final_model_submission.ipynb
```

to regenerate the final model artifacts locally.

## Reusable Source Modules

Reusable project logic is organised in the `src/` directory.

### `src/preprocessing.py`

Contains:

- feature-type detection;
- tree-model preprocessing;
- scaled preprocessing.

### `src/train_models.py`

Contains model-building functions for:

- Logistic Regression;
- Random Forest;
- Linear SVM;
- MLP Neural Network.

### `src/evaluate_models.py`

Contains:

- Accuracy calculation;
- Balanced Accuracy calculation;
- Macro Precision;
- Macro Recall;
- Macro F1;
- Classification Report;
- Confusion Matrix utilities.

### `src/predict.py`

Contains:

- saved-model loading;
- single-record prediction;
- competition submission generation.

## Outputs

Important generated outputs include:

```text
outputs/all_model_comparison.csv
outputs/best_model_per_family.csv
outputs/submissions/submission.csv
outputs/submissions/submission_preview.csv
outputs/metrics/final_submission_metadata.json
```

Important visual evidence includes:

- confusion matrices;
- model-comparison charts;
- Balanced Accuracy ranking;
- best-model family comparison;
- Random Forest feature importance;
- MLP model comparison.

## Limitations

- The project uses competition data and should not be treated as a clinical diagnostic system.
- Predictions depend on the features and patterns represented in the training dataset.
- Feature importance indicates predictive contribution rather than causation.
- A Linear SVM was used instead of a full non-linear kernel SVM due to the large dataset size and computational constraints.
- Validation performance does not guarantee identical leaderboard performance.

## Ethical Considerations

Health-related predictions can be sensitive. Therefore:

- the application is presented for educational and research purposes;
- predictions should not replace professional medical assessment;
- model outputs should not be used to stigmatise students;
- confidence scores should not be interpreted as medical certainty;
- human oversight is necessary for any real-world decision-making.

## Evidence for Assessment and Demonstration

The final report and demonstration should include:

- competition page evidence;
- competition start and end dates;
- official evaluation metric;
- public leaderboard submission screenshot;
- final/private submission screenshot when available;
- model comparison table;
- best-model confusion matrix;
- classification report;
- feature-importance chart;
- Streamlit application screenshots;
- GitHub repository evidence.

## Author

Undergraduate Computational Intelligence Project

## Disclaimer

This project is intended for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment.
