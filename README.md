# How Likely Are You to Be Admitted to a Hospital?

## TL;DR:

- ML pipeline using CMS synthetic Medicare claims data
- Engineered temporal and clinical features from longitudinal claims records
- Compared Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost models
- Explained model predictions using SHAP
- Tracked experiments and artifacts with MLflow

# Final Report: 30-Day Hospital Admission Risk Prediction Using CMS Synthetic Claims Data

## Abstract

Hospital readmission is an important quality and cost signal in healthcare, and predicting which patients are at elevated risk can assist in earlier intervention and better resource planning. This project provides an end-to-end ML pipeline that uses CMS synthetic Medicare claims data to classify patients based on the probability that they will be admitted to a hospital within 30 days. (How does the feature data avoid leaking info about the target variable?). It includes data cleaning, feature engineering, predictive modeling, model explainability (SHAP), and experiment tracking (MLflow).

Because hospital admission within 30 days is a relatively rare outcome, the dataset classes are imbalanced (with a positive class prevalence of 4.3%). Imbalanced data tends to render accuracy -- that is, (TP+TN) / (TP+TN+FP+FN) -- as an insufficient evaluation metric on its own, so ROC-AUC and Precision-Recall AUC (with more of an emphasis on recall) tended to be more informative evaluation metrics. The final CatBoost model achieved ROC-AUC of 0.892 and Precision-Recall AUC of 0.393.

## Introduction

Predicting hospital readmission is a common healthcare analytics problem because it has direct implications for care coordination, population health management, and operational efficiency. Patients who are likely to return shortly after discharge may benefit from closer follow-up, medicaiton reconciliation, discharge planning, or additional support services. From a data science perspective, readmission prediction is also challenging because the data is often longitudinal, sparse, imbalanced, and encoded using industry-specific billing and diagnosis systems.

This project builds a reproducible admission prediction pipeline using synthetic CMS claims data, with an emphasis on data preparation, feature engineering, model comparison, and explainability. The project aligns closely with common responsibilities in healthcare and biotech data science roles, including preparing AI-ready datasets, training machine learning models, evaluating performance, and documenting results as clearly as possible.

## Dataset and Prediction Target

The analysis uses CMS synthetic claims data, including beneficiary-level demographic and chronic condition information as well as inpatient claim records. Beneficiary data provides demographic variables such as sex, race, age, state, and chronic condition indicators. Inpatient claims provide hospitilization details such as admission and discharge dates, diagnosis codes, procedure codes, and utilization measures.

The prediction target is binary:

1: the patient is admitted within 30 days of discharge
0: the patient is not admitted within 30 days

The target was derived by sorting admissions chronologically at the beneficiary level and measuring the number of days between a discharge and the next admission. A readmission label was assigned when the next admission occurred within 30 days. Negative intervals were treated as invalid and excluded from the label generation logic. (How were instances of people who were being admitted for the first time handled?) The final modeling dataset contained a positive class prevalence of 4.3%.

## Data Cleaning

First, the beneficiary and inpatient files were cleaned separately. Rows with missing beneficiary identifiers were removed, duplicate rows were dropped, and date fields were converted into proper datetime format. Records with impossible date relationships were filtered out, such as discharge dates earlier than admission dates or birth dates inconsistent with death dates. These checks were important because claims data can contain irregularities that would otherwise introduce outlying noise.

Next, the beneficiary and inpatient datasets were merged on the beneficiary identifier. This created a patient-hospitalization table that combined demographic context, chronic condition information, and claim-level utilization details. This harmonized table formed the basis for subsequent feature engineering and modeling.

## Feature Engineering

Several feature families were constructed to capture patient history, hospitalization burden, and clinical complexity.

Temporal variables were derived to reflect the longitudinal nature of claims data and capture utilization intensity and recency. These included:

days_since_last_admission
prior_admissions
admissions_last_7d
admissions_last_14d
admissions_last_30d
admissions_last_90d
admissions_last_180d

Additional engineered variables included:

length_of_stay
clm_utlztn_day_cnt
age
comorbidity_count

Chronic disease indicators for CHF, diabetes, COPD, kidney disease, cancer, depression, etc. were used to create a column called "comorbidity_count" which, as the name intuitively implies, contains the number of comorbidities a patient possesses.

Because raw ICD-9 diagnosis and procedure codes are sparse and have high cardinality, those fields were grouped into broader categories to create more comprehensible summary features. Specifically, diagnosis codes were mapped into the following groups: 

cardiovascular
respiratory
endocrine
kidney
other 

and procedure codes were mapped into the following groups:

nervous system
endocrine
ear/nose/throat
cardiovascular
digestive
urinary
musculoskeletal
other

## Modeling Strategy

The modeling process was designed to reflect best practices for imbalanced classification and healthcare data evaluation.

The evaluated models included:

Logistical Regression
Random Forest
XGBoost
LightGBM
CatBoost

Numerical features were standardized and medians were imputed where there were missing values. Categorical features were one-hot encoded where appropriate (obviously categorical features did not need to be preprocessed for CatBoost).

Because the dataset is heavily imbalanced, the primary evaluation metrics were:

ROC-AUC
Precision-Recall AUC
Recall
Precision
F1-score

As mentioned previously in the abstract, accuracy was not used as the main evaluation metric because with imbalanced data, a model can achieve high accuracy by simply predicting the majority class most of the time, so recall and PR-AUC are more informative metrics.

Model selection was based on 5-fold grouped stratified cross-validation at the beneficiary level, followed by holdout evaluation, to ensure that admissions from the same patient were kept within a single fold, preventin gpatient-level information leakage across training and validation splits. This grouped split strategy helps ensure that reported performance is a realistic estimate of generalization to unseen patients. On the held-out test set, the final CatBoost model achieved ROC-AUC of 0.892 and a Precision-Recall AUC of 0.393. 

These results are meaningful given the low base rate of readmission. A PR-AUC of 0.393 is substantially above the 4.3% prevalence baseline and indicates that the model captures a decent amount of signal beyond chance in a highly imbalanced and ostensibly noisy setting.

It should be noted that because Precision-Recall AUC was the primary evaluation metric, there was an opportunity to adjust the model's decision threshold. The model's decision threshold (0.87) was optimized for high recall (0.82), prioritizing the identification of as many patients at risk of readmission as possible, even at the expense of additional false positives. In a healthcare risk setting, missing high-risk patients is more consequential than issuing some false positive alerts, so when in doubt, it's better to overdiagnose: better safe than sorry.

SUGGESTED FIGURE INSERTIONS:

- ROC curve for final CatBoost model
- Precision-Recall curve for final CatBoost model
- Confusion matrix at the chosen decision threshold
- Table summarizing cross-validation and holdout metrics for all models.

## Explainability

Talk about:

- Global feature importance summaries
- Feature attribution plots
- Patient-level explanations for representative examples
- Comparisons between true positives, false positives, false negatives, and true negatives

Answer:

- Which features most strongly influence readmission risk?
- Do recent admissions and utilization history matter more than demographics?
- Are the model's explanations plausible?

SUGGESTED FIGURE INSERTIONS:

- SHAP summary plot
- SHAP bar plot for global feature importance
- SHAP waterfall plot for arepresentative patient
- Example explanations for one false positive and one false negative

## Experiment Tracking and Reproducibility

To make the project more realistic and maintainable, experiment trackingwas incorporated using MLflow. This allows model metrics, parameters, and artifacts to be logged consistently across runs. The model also saves model artifacts and metadata, including hte final model file, cross-validation results, holdout metrics, and model configuration information. Maybe there's something to talk about here?

## Limitations

This project has several limitations. First, the data is synthetic, so the results should not be interpreted as a clinical prediction tool. Second, claims data do not contain rich real-time clinical signals such as vitals, lab trends, or notes, which may limit predictive power. Third, readmission is influenced by many external factors, including care access, socioeconomic context, and hospital-specific discharge practices, which are only partially represented in claims data. Finally, performance may vary across different target definitions, feature sets, and splitting strategies.

## Conclusion

This project demonstrates a complete healthcare machine learning workflow using CMS synthetic claims data. Starting from raw beneficiary and inpatient files, the project cleaned and harmonized longitudinal claims records, engineered temporal and utilization features, trained and compared multiple machine learning models, and evaluated them using metrics appropriate for an imbalanced outcome. The final CatBoost model achieved a ROC-AUC of 0.892 and a PR-AUC of 0.393 on a dataset with 4.3% positive prevalence. Beyond model performance, the project emphasizes reproducibility, patient-level data leakage prevention, explainability, and experiment tracking.