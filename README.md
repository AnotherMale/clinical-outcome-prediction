# 30-Day Hospital Readmission Risk Prediction Using CMS Synthetic Claims Data

## TL;DR

- ML pipeline using CMS synthetic Medicare claims data
- Engineered temporal and clinical features from longitudinal claims records
- Compared Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost models
- Explained model predictions using SHAP
- Tracked experiments and artifacts with MLflow

An end-to-end ML pipeline that uses CMS synthetic Medicare claims data to classify patients based on the probability that they are readmitted to a hospital within 30 days of being discharged. Includes data cleaning, feature engineering, predictive modeling, model explainability (SHAP), and experiment tracking (MLflow).

The final CatBoost model achieved a ROC-AUC of 0.892 and a Precision-Recall AUC of 0.393 on a highly imbalanced dataset with a 4.3% 30-day readmission rate.

Because hospital readmissions are relatively rare, accuracy ( (TP+TN) / (TP+TN+FP+FN) ) is not a sufficiently informative performance metric. Instead, the model threshold (0.087) was optimized to prioritize high recall (0.82), prioritizing the identification of as many patients at risk of readmission as possible, even at the expense of additional false positives.

Models were assessed using 5-fold Stratified Group Cross-Validation at the beneficiary level. All admissions from the same beneficiary were assigned to a single fold, preventing patient-level information leakage between training and validation sets. Performance was evaluated primarily using ROC-AUC and Precision-Recall AUC, since those metrics are better suited for highly imbalanced data.