# Cardiovascular Disease Prediction Project

## Target Variables
- **CVDINFR3** → Myocardial Infarction (Heart Attack)
- **CVDCRHD3** → Coronary Heart Disease
- **CVDSTRK3** → Stroke

## Problem Type
- Binary or Multiclass Classification
- The goal is to predict whether an individual has a history of cardiovascular events based on health and lifestyle data.

## Objective
Predict the probability of an individual having a history of:
- **Myocardial Infarction (CVDINFR3)**
- **Coronary Heart Disease (CVDCRHD3)**
- **Stroke (CVDSTRK3)**

## Relevance
These health outcomes are directly associated with key lifestyle and clinical factors, such as:
- **BMI (Body Mass Index)**
- **Physical Activity** → (_TOTINDA)
- **Smoking Habits** → (SMOKE100, SMOKDAY2)
- **Alcohol Consumption** → (DRNKANY4)
- **Diabetes Diagnosis** → (DIABETE2)

This project supports preventive care strategies by accurately predicting cardiovascular risks and enables early interventions for at-risk individuals.

## Data Cleaning and Preprocessing
### Correction of Weight, Height, and BMI values.

### Class Balancing:
- The dataset had a significant class imbalance with far more negative cases than positive ones for each target.
- To address this, the datasets were balanced by reducing the number of negative cases to match the positive ones for each target.

### Variable Refinement:
- For variables like **SMOKE100, DRNKANY5, and _TOTINDA**, responses coded as **7 ("Don't know")** or **9 ("Refused")** were removed to ensure data quality.

### Diabetes Variable Transformation (Based on Codebook):
| Code | Meaning | Transformed Value |
|------|---------|------------------|
| 1 | Yes | Yes |
| 2 | Yes, but only during pregnancy | No |
| 3 | No | No |
| 4 | Prediabetes | Yes |
| 7 | Don't know | Removed |
| 9 | Refused | Removed |

### Binary Encoding:
- All categorical variables were converted to binary (0 and 1) for compatibility with machine learning models.

## Model Training and Evaluation
### Target: **CVDINFR4**
- **Accuracy**: 77.56%
- **AUC-ROC**: 0.81
- **Classification Report**:
  - Class 0 (majority): High precision and recall.
  - Class 1 (minority): Moderate precision and recall.
- **Insight**: The stacking model performs decently but has room for improvement, particularly in detecting minority cases.

### Target: **CVDCRHD4**
- **Accuracy**: 77.16%
- **AUC-ROC**: 0.80
- **Classification Report**:
  - Similar to CVDINFR4, class 1 shows lower recall and f1-score, suggesting under-detection of minority cases.

### Target: **CVDSTRK3**
- **Accuracy**: 78.77%
- **AUC-ROC**: 0.67
- **Classification Report**:
  - The model struggles significantly with class 1, evident from a recall of only 3% and a very low f1-score.
  - Although overall accuracy is high, it is mainly due to the model focusing on the majority class (class 0).

## Model Optimization Techniques
**Objective**: Improve model performance by tuning hyperparameters and adjusting the data.
### Approach:
- Applied **Random Search** for hyperparameter optimization.
- Evaluated the best model configurations.
- Extracted **feature importance** from optimized models.

## Advanced Modeling: Stacking and SHAP Analysis
- Implemented **Stacking Models** to combine predictions from multiple base models for better generalization.
- To interpret feature importance in the **StackingClassifier**, applied **SHAP (SHapley Additive exPlanations)** as the model doesn’t have a direct `feature_importances_` attribute.

### Why SHAP?
- SHAP values offer insights into how each feature contributes to the final prediction, making it an effective tool for interpreting complex ensemble models like stacking.

## How to Run the Script
### Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install Dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

### Run the Python Script:
```bash
python Proyecto_2_1.py
```

## Folder Structure
```
.
├── Proyecto_2_1.py          # Main analysis script
├── data/                    # Folder containing datasets
│   └── [your-dataset-files]
└── README.md                 # Project documentation
```

## Future Improvements
- Expand data analysis with advanced statistical tests.
- Integrate data visualizations using **Plotly** or **Dash**.
- Optimize data pipelines for larger datasets.

---
**Author:** Federico Trotta
**Date:** 25/2/2025

