# 30-Day Hospital Readmission Risk Prediction Using CMS Synthetic Claims Data

Developed a readmission risk model using CMS synthetic claims data. The final XGBoost model achieved ROC-AUC 0.884 and PR-AUC 0.388 on a dataset with only 4.3% positive prevalence, demonstrating strong discrimination of high-risk patients.

For this project, a good recall score matters more than a good accuracy score, because missing a high-risk patient is usually more dangerous than a false positive.

Models were evaluated using 5-fold grouped stratified cross-validation at the beneficiary level. Admissions from the same beneficiary were restricted to a single fold to prevent information leakage. Performance was assessed using ROC-AUC and Precision-Recall AUC due to the highly imbalanced readmission outcome (4.3% prevalence).