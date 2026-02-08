# Credit-Risk
Credit Risk Prediction Model
# Project overview

This project focuses on predicting credit default risk using machine learning models.
The goal was to build a classification model that helps identify high-risk clients.

# Dataset

UCI Default of Credit Card Clients dataset
Contains financial and behavioral information about credit card clients.

# Steps

Exploratory Data Analysis (EDA)

Handling imbalanced data using SMOTE

Model building:

Logistic Regression

Random Forest

Model evaluation (ROC-AUC, confusion matrix)

Model interpretability using SHAP

# Results

ROC-AUC ≈ 0.75

Improved default detection after SMOTE

Key risk factors:

repayment history

credit limit

previous payment amounts

 # Business value

The model can support:

credit approval decisions

risk segmentation

early warning systems

Variable Name	Role	Type	Demographic	Description	Units	Missing Values
ID	ID	Integer				no
X1	Feature	Integer		LIMIT_BAL		no
X2	Feature	Integer	Sex	SEX		no
X3	Feature	Integer	Education Level	EDUCATION		no
X4	Feature	Integer	Marital Status	MARRIAGE		no
X5	Feature	Integer	Age	AGE		no
X6	Feature	Integer		PAY_0		no
X7	Feature	Integer		PAY_2		no
X8	Feature	Integer		PAY_3		no
X9	Feature	Integer		PAY_4		no
X10	Feature	Integer		PAY_5		no
X11	Feature	Integer		PAY_6		no
X12	Feature	Integer		BILL_AMT1		no
X13	Feature	Integer		BILL_AMT2		no
X14	Feature	Integer		BILL_AMT3		no
X15	Feature	Integer		BILL_AMT4		no
X16	Feature	Integer		BILL_AMT5		no
X17	Feature	Integer		BILL_AMT6		no
X18	Feature	Integer		PAY_AMT1		no
X19	Feature	Integer		PAY_AMT2		no
X20	Feature	Integer		PAY_AMT3		no
X21	Feature	Integer		PAY_AMT4		no
X22	Feature	Integer		PAY_AMT5		no
X23	Feature	Integer		PAY_AMT6		no
Y	Target	Binary		default payment next month
