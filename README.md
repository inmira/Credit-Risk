# Credit Card Default Prediction

## Project Overview

This project develops and evaluates machine learning models to predict credit card payment default using the **UCI Default of Credit Card Clients** dataset.

The analysis focuses on model comparison, class imbalance, threshold selection, and model interpretability. Four approaches are evaluated: Logistic Regression, Random Forest, Random Forest with SMOTE, and LightGBM.

## Dataset

- **Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Observations:** 30,000 clients
- **Features:** 23 explanatory variables
- **Target:** `DEFAULT` — 1 indicates default; 0 indicates no default
- **Default rate:** 22.12%

The input variables describe credit limits, demographic information, repayment status, bill statements, and previous payment amounts.

For readability, the original feature labels (`X1`–`X23`) are mapped to descriptive names such as `LIMIT_BAL`, `PAY_0`, `BILL_AMT1`, and `PAY_AMT1`.

## Methodology

### Data preparation
- Loaded the dataset from the UCI repository.
- Renamed the explanatory variables using the dataset documentation.
- Checked target distribution and missing values.
- Used a stratified 80/20 train–test split.

### Models
1. Logistic Regression
2. Random Forest
3. Random Forest with SMOTE
4. LightGBM

Median imputation is included in the model pipelines. Logistic Regression additionally uses `StandardScaler`.

### Validation and class imbalance
- Applied 5-fold `StratifiedKFold` cross-validation on the training set.
- Generated out-of-fold (OOF) probabilities for model comparison and threshold selection.
- Integrated SMOTE into an `imblearn` pipeline, so oversampling is performed only on the training portion of each cross-validation fold.
- Selected an F1-based classification threshold using OOF training predictions. The test set was not used for threshold selection.
- Evaluated the final fitted models on the untouched test set.

### Evaluation metrics
ROC-AUC and Average Precision evaluate probability ranking. Precision, recall, and F1 evaluate classification performance at a selected threshold. Recall for class 1 is particularly relevant when assessing how many default cases are identified.

## Model Performance

### Cross-validation results

Out-of-fold results on the training data:

| Model | OOF ROC-AUC | OOF Average Precision |
|---|---:|---:|
| LightGBM | **0.7816** | **0.5574** |
| Random Forest | 0.7742 | 0.5502 |
| Random Forest + SMOTE | 0.7733 | 0.5431 |
| Logistic Regression | 0.7259 | 0.5042 |

LightGBM achieved the highest OOF ROC-AUC and Average Precision among the evaluated models.

### Test-set results

The following metrics use each model's threshold selected from OOF training predictions:

| Model | Threshold | ROC-AUC | Average Precision | Precision (default) | Recall (default) | F1 (default) |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.3084 | **0.7780** | **0.5512** | 0.5353 | **0.5365** | **0.5359** |
| Random Forest + SMOTE | 0.3788 | 0.7652 | 0.5433 | 0.5279 | 0.5275 | 0.5277 |
| Random Forest | 0.3448 | 0.7641 | 0.5461 | **0.5498** | 0.5079 | 0.5280 |
| Logistic Regression | 0.2864 | 0.7076 | 0.4939 | 0.5319 | 0.4770 | 0.5030 |

LightGBM had the strongest test ROC-AUC, Average Precision, recall, and F1 among these models at their selected thresholds. Random Forest had the highest default-class precision.

### Random Forest and SMOTE at the default threshold

To separate the effect of resampling from threshold selection, Random Forest models were also compared at a fixed threshold of 0.50:

| Model | ROC-AUC | Precision (default) | Recall (default) | F1 (default) |
|---|---:|---:|---:|---:|
| Random Forest | 0.7641 | **0.6536** | 0.3640 | 0.4676 |
| Random Forest + SMOTE | 0.7652 | 0.6151 | **0.4107** | **0.4925** |

At the 0.50 threshold, SMOTE increased default recall from **0.3640 to 0.4107**, while precision decreased from **0.6536 to 0.6151**. In this run, ROC-AUC changed only slightly, from 0.7641 to 0.7652.

This result illustrates the trade-off between identifying more default cases and generating more false positive predictions. SMOTE should therefore not be described as an unconditional improvement.

## Model Interpretation

Feature importance was examined for the LightGBM model, selected as the best tree-based model by OOF ROC-AUC.

The highest-ranked features by LightGBM split importance included:

1. `BILL_AMT1`
2. `PAY_AMT3`
3. `LIMIT_BAL`
4. `AGE`
5. `PAY_AMT5`
6. `PAY_AMT1`
7. `PAY_AMT2`
8. `PAY_AMT6`
9. `BILL_AMT2`
10. `PAY_AMT4`

SHAP `TreeExplainer` was used to examine feature contributions for a sample of test observations. The notebook includes a SHAP summary plot with descriptive feature names.

Feature importance and SHAP values describe how the fitted model uses the input variables; they do not establish causal relationships.

## Key Findings

- **LightGBM performed best overall** in this experiment, achieving a test ROC-AUC of 0.7780 and Average Precision of 0.5512.
- **Threshold selection matters.** A threshold below 0.50 was selected for every model when maximizing F1 on OOF training predictions.
- **SMOTE changed the precision–recall balance.** At a fixed threshold of 0.50, it increased recall for defaults but reduced precision.
- **Random Forest and LightGBM outperformed Logistic Regression** in ROC-AUC on both the OOF training predictions and the held-out test set.
- **Repayment and bill-related variables** were prominent in the LightGBM feature-importance ranking.

## Repository Contents

- `UCI_Credit_Card_Default_Prediction.ipynb` — complete analysis, model training, evaluation, and interpretation.
- `requirements.txt` — Python dependencies.

## How to Run

The notebook can be opened in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.

1. Open the `.ipynb` file.
2. Run the installation cell.
3. Run the notebook cells from top to bottom.

The dataset is fetched directly from the UCI Machine Learning Repository.

## Tools and Libraries

Python · pandas · NumPy · scikit-learn · imbalanced-learn · LightGBM · SHAP · Matplotlib


