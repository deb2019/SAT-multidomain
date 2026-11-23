# Data

The datasets are **not** stored in this repository (several licences do not allow redistribution, and the card-transaction files are larger than GitHub's file limit). Download each file from its source, put it anywhere inside `data/` (sub-folders are fine), keep the exact file name, and run

```bash
python scripts/check_data.py .
```

to confirm by SHA-256 checksum (`checksums.sha256`) that you have the same files as the paper. If a file differs, the scripts still run but numbers will not match exactly.

## Controlled-shift benchmark (32 datasets)

The positive class is the **minority value of the target column** (attack, malware, fraud, failure, death, etc.). Columns listed in `drop_columns.json` (identifiers, post-outcome variables, failure-mode flags and row indices) are removed automatically before modelling.

| ID | Dataset | Domain | File | Target column | Licence | Source |
|---|---|---|---|---|---|---|
| C01 | Malware (static features) | Cyber security | `imbalanced_malware_dataset.csv` | `Label` | not recorded | source to be added |
| C04 | Cyber intrusion (IDS) | Cyber security | `Cybersecurity_Preprocessed_Imbalanced.csv` | `Label` | not recorded | source to be added |
| D001 | ClaMP malware (PE headers) | Cyber security | `ClaMP_Imbalanced.csv` | `class` | CC BY 4.0 | https://www.kaggle.com/datasets/saurabhshahane/classification-of-malwares |
| D002 | Link phishing | Cyber security | `pre_link_phishing_imb.csv` | `status` | MIT License | https://www.kaggle.com/datasets/winson13/dataset-for-link-phishing-detection/data |
| D004 | Android malware | Cyber security | `AndroidMalware_Imbalanced.csv` | `Result` | CC0 1.0 Universal | https://www.kaggle.com/datasets/ankit1743/android-malware-detection-dataset |
| D005 | Network intrusion (web attacks) | Cyber security | `Cybersecurity_WebAttack_Preprocessed_30k.csv` | `Label` | CC0 1.0 Universal | https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset |
| D006 | Malicious URL | Cyber security | `Phishing_Imb_Pre.csv` | `label` | Apache 2.0 | https://www.kaggle.com/datasets/pilarpieiro/tabular-dataset-ready-for-malicious-url-detection/data |
| D021 | Loan approval | Finance & fraud | `ready_loan_data.csv` | `loan_status` | MIT License | https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset |
| D039 | Insurance claim fraud | Finance & fraud | `pre_ins_fraud.csv` | `fraud_reported` | CC BY 4.0 | https://data.mendeley.com/datasets/992mh7dk9y/2 |
| D042 | Caravan insurance (COIL 2000) | Finance & fraud | `caravan_preprocessed.csv` | `CARAVAN` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000 |
| D044 | Telco churn | Finance & fraud | `pre_telco_churn.csv` | `Churn` | CC0 1.0 Universal | https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/input |
| D046 | Bank marketing | Finance & fraud | `pre_bank_marketing.csv` | `y` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| D048 | Iranian churn | Finance & fraud | `customer_churn_preprocessed.csv` | `Churn` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |
| D049 | Online shoppers | Finance & fraud | `online_shoppers_preprocessed.csv` | `Revenue` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset. |
| D050 | Airbnb listing fraud | Finance & fraud | `preprocessed_airbnb_fraud.csv` | `fraud` | AGPL-3.0 (GNU Affero General Public License, version 3) | https://www.kaggle.com/datasets/hawkingcr/airbnb-for-boston-with-fraud-detection |
| D054 | Vehicle claim fraud | Finance & fraud | `preprocessed_training_data.csv` | `fraud` | Unspecified (see source) | https://www.kaggle.com/datasets/surekharamireddy/fraudulent-claim-on-cars-physical-damage |
| D056 | Insurance response | Finance & fraud | `preprocessed_insu_response.csv` | `Response` | CC0 1.0 Universal | https://www.kaggle.com/datasets/arashnic/imbalanced-data-practice |
| D075 | Machine failure (AI4I 2020) | Industrial & anomaly | `ai4i2020_preprocessed.csv` | `Machine failure` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| D076 | Annealing | Industrial & anomaly | `Annealing_Binary_Preprocessed.csv` | `Label` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/3/annealing |
| D077 | APS failure (Scania trucks) | Industrial & anomaly | `aps_failure_reduced_30k.csv` | `class` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks |
| D079 | Pulsar candidates (HTRU2) | Industrial & anomaly | `HTRU_2_preprocessed.csv` | `Class` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/372/htru2 |
| D058 | Alzheimer's disease | Medical | `pre_alzheimers_disease.csv` | `Diagnosis` | CC BY 4.0 | https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset |
| D059 | Breast cancer status | Medical | `Breast_Cancer_Preprocessed.csv` | `Status` | Apache 2.0 | https://www.kaggle.com/datasets/marshalpatel3558/breast-cancer-prediction |
| D060 | Diabetes indicators (BRFSS) | Medical | `preprocessed_diabetes_data.csv` | `Diabetes_binary` | CC BY 4.0 | https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset |
| D061 | Healthy lifestyle | Medical | `Healthy_Lifestyle_Imbalanced.csv` | `Label` | CC0 1.0 Universal | https://www.kaggle.com/datasets/aditibabu/healthy-lifestyle |
| D062 | Hospital length of stay | Medical | `LengthOfStay_Preprocessed.csv` | `long_stay` | MIT License | https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft |
| D063 | ICU mortality | Medical | `ICU_Death_Prediction_Preprocessed.csv` | `target` | CC BY-NC-SA 4.0 | https://www.kaggle.com/datasets/fdemoribajolin/death-classification-icu |
| D064 | NHS admissions | Medical | `imbalanced_nhs_dataset.csv` | `Target_High_Admission` | MIT License | https://www.kaggle.com/datasets/tomfarnell/nhs-admissions-data-2011-2020 |
| D069 | Child stunting | Medical | `preprocessed_stunting_sampled_10k.csv` | `Stunting_binary` | Apache 2.0 | https://www.kaggle.com/datasets/jabirmuktabir/stunting-wasting-dataset |
| D070 | SUPPORT2 mortality | Medical | `support2_preprocessed_binary.csv` | `target` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/880/support2 |
| D071 | Cardiotocography (CTG) | Medical | `CTG_preprocessed_scaled.csv` | `NSP` | CC BY 4.0 | https://archive.ics.uci.edu/dataset/193/cardiotocography |
| D081 | ECG arrhythmia (MIT-BIH) | Medical | `MITBIH_ECG_Binary_Preprocessed.csv` | `Label` | Open Data Commons Attribution License (ODC-By) v1.0 | https://www.kaggle.com/code/redpen12/anomaly-detection-with-cnn-autoencoders/input |

## Temporal case studies

| Case | Files | Source |
|---|---|---|
| Simulated card transactions (Sparkov) | `fraudTrain.csv`, `fraudTest.csv` | https://www.kaggle.com/datasets/kartik2112/fraud-detection |
| Bank marketing | `pre_bank_marketing.csv` (also in the benchmark; rows in calendar order) | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Diabetes readmission | `diabetic_data.csv` | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |

`sources.csv` lists every candidate dataset that was audited (75 entries), with source, licence and notes; `audit_final.csv` records the audit and the include/exclude decision for each.
