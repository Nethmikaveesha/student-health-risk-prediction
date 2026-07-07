
# Student Health Risk Prediction


## Project Aim

This project predicts student health condition using health and lifestyle-related indicators. Four machine learning and computational intelligence models were developed and compared.

## Models Used

1. Logistic Regression
2. Random Forest
3. Support Vector Machine
4. MLP Neural Network

## Best Model

The best validation result was achieved by the **Tuned Random Forest** model.

| Model Family | Best Variant | Balanced Accuracy |
|---|---|---:|
| Random Forest | Tuned Random Forest | 0.945314 |
| MLP Neural Network | Tuned MLP Neural Network | 0.945256 |
| Logistic Regression | Class-Weighted Logistic Regression | 0.898900 |
| SVM | Class-Weighted Linear SVM | 0.881395 |

## Project Structure

```text
app/
data/
models/
notebooks/
outputs/
report/
src/
requirements.txt
README.md
