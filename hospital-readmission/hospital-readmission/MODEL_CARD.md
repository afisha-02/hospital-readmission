# Model Card: Hospital Readmission Risk Intelligence

**Model Version:** 1.0.0  
**Date of Release:** November 2026  
**Type:** Calibrated Gradient Boosted Decision Tree (XGBoost Classifier)  
**Task:** 30-Day Unplanned Hospital Readmission Risk Estimation  
**License:** MIT License  
**Maintainer / Authors:** Advanced ML Engineering Team  

---

## 1. Model Summary
Hospital Readmission Risk Intelligence is an explainable machine learning platform engineered to estimate the mathematical probability of a patient experiencing an unplanned hospital readmission within 30 days of discharge. The system features:
- Scikit-learn preprocessing with strict train/test data leakage prevention.
- Optuna Bayesian-optimized XGBoost classifier utilizing `scale_pos_weight` loss penalization.
- Post-hoc Platt/Sigmoid probability calibration yielding a **63.4% reduction in Brier score error** (Brier = 0.0796, ECE = 0.0064).
- Cost-sensitive decision threshold optimization operating at an F1/Cost-optimal cutoff of **0.125** instead of naive 0.50.
- Native TreeSHAP explainability providing patient-level waterfall factor attributions.
- Population Stability Index (PSI) and Kolmogorov-Smirnov continuous statistical drift monitoring.

---

## 2. Intended Use
- **Primary Intended Use:** Analytical risk estimation, care coordinator decision support, and retrospective clinical quality benchmarking to prioritize post-discharge nurse outreach calls and medication reconciliation reviews.
- **Intended Users:** Healthcare data scientists, clinical quality improvement committees, care coordinators, and health systems researchers.
- **Operating Environment:** Deployed via FastAPI REST service and interactive Streamlit web dashboard.

---

## 3. Out-of-Scope & Prohibited Use
> [!CAUTION]
> **RESEARCH & PORTFOLIO PROTOTYPE ONLY**  
> This platform has **NOT** undergone prospective randomized clinical trials and is **NOT** a certified Medical Device (e.g., FDA 510(k), SaMD, CE mark).

- **Prohibited:** Autonomous discharge decisions, bed allocation, or denying patient admission based on risk scores.
- **Prohibited:** Replacing direct bedside clinical diagnosis, physician clinical judgment, or patient physical examinations.
- **Prohibited:** Deriving causal biological claims (feature attributions denote statistical weights, never physiological causation).

---

## 4. Dataset & Data Provenance
- **Source:** UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years 1999–2008* (Strack et al., 2014).
- **Initial Encounters:** 101,766 hospitalizations across 130 US hospitals.
- **Data Cleansing & Leakage Prevention:**
  1. **Terminal / Hospice Exclusion:** 2,423 encounters (2.38%) with discharge codes 11 (Expired) and 13, 14, 19, 20, 21 (Hospice) were filtered out because in-hospital mortality precludes readmission and distorts survival curves.
  2. **Patient Deduplication (Primary Encounter Policy):** 29,353 repeat encounters were eliminated, retaining strictly 1 primary encounter per unique patient (`patient_nbr`). This guarantees zero longitudinal patient leakage across training, validation, and test sets.
  3. **High Missingness Pruning:** Features missing >30% of values (`weight`: 97%, `medical_specialty`: 49%, `payer_code`: 40%) were excluded from training to prevent imputation hallucination.
- **Evaluated Dataset:** 69,990 clean patient encounters (88 encoded features after ColumnTransformer encoding).
- **Target Formulation:** Binary classification where positive class (1) = Readmission within 30 days (`<30`), representing **8.98% prevalence (6,285 positive cases)**. Negative class (0) = No readmission or readmission after 30 days (63,705 cases).

---

## 5. Feature Space
- **Utilization Features (Past 12 Months):** `number_inpatient`, `number_emergency`, `number_outpatient`, `total_encounters_prior`, `utilization_intensity`.
- **Inpatient Acuity Features:** `time_in_hospital` (length of stay), `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`, `lab_per_day`, `med_per_day`.
- **Clinical & Diagnostic Features:** Primary and secondary ICD-9 mapped diagnostic organ systems (`Circulatory`, `Respiratory`, `Digestive`, `Diabetes`, `Injury/Poisoning`, `Musculoskeletal`, `Genitourinary`, `Neoplasms`, `Other`).
- **Pharmacotherapy & Glycemic Control:** `max_glu_serum`, `A1Cresult`, `change` (regimen titrated during stay), `diabetesMed` (prescribed), `insulin`, `metformin`, `diabetes_med_count`.
- **Demographics & Administration:** `age` bracket, `gender`, `race`, `admission_type_desc`, `discharge_disposition_desc`.

---

## 6. Training & Preprocessing
- **Partition:** Stratified 70% Train (48,993), 15% Validation (10,498), 15% Test (10,499).
- **Pipeline:** Scikit-learn `ColumnTransformer` fitted strictly on `X_train`:
  - Numerical: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - Categorical: `SimpleImputer(strategy='constant', fill_value='Missing')` followed by `OneHotEncoder(handle_unknown='ignore')`.
- **Optimization:** Optuna Bayesian search maximizing validation PR-AUC with early stopping (20 rounds).
- **Imbalance Handling:** XGBoost loss function weighting via `scale_pos_weight = 10.13` (negative-to-positive ratio).

---

## 7. Quantitative Benchmark Evaluation
Evaluated on **10,499 held-out patient encounters**:

| Model Candidate | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 0.0898 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0818 |
| **Logistic Regression (Balanced)** | 0.1630 | 0.6375 | 0.5217 | 0.1340 | 0.2126 | 0.2318 |
| **Random Forest (Balanced Subsample)** | 0.1573 | 0.6340 | 0.3839 | 0.1517 | 0.2176 | 0.2067 |
| **XGBoost (Uncalibrated, 0.50)** | 0.1639 | 0.6372 | 0.5037 | 0.1388 | 0.2183 | 0.2174 |
| **XGBoost (Calibrated, 0.125 Cutoff)** | **0.1639** | **0.6372** | **0.3499** | **0.1635** | **0.2230** | **0.0796** |

---

## 8. Probability Calibration
- Raw gradient boosting probabilities trained with `scale_pos_weight` produce uncalibrated log-odds shifted towards 0.5-0.7.
- Post-hoc Platt/Sigmoid calibration maps model outputs to true frequentist probabilities matching the ~9% base rate.
- **Uncalibrated:** Brier Score = 0.2174, Expected Calibration Error (ECE) = 0.3622.
- **Calibrated:** Brier Score = **0.0796**, Expected Calibration Error (ECE) = **0.0064** (**63.4% error reduction**).

---

## 9. Decision Threshold & Cost Economics
- Naive 0.50 cutoff fails on imbalanced healthcare datasets (5.5x higher than baseline readmission rate).
- **F1-Optimal Threshold:** **0.125** (Recall = 34.99%, Precision = 16.35%, F1 = 0.2230).
- **Cost-Optimal Threshold:** **0.100** ($394.61 expected cost/patient).
- Under an illustrative cost matrix ($C_{FN}=\$5,000, C_{FP}=\$500, C_{TP}=\$750$), operating at the cost-optimal cutoff saves **\$572,000 (12.1% reduction in total decision burden)** compared to naive 0.50.

---

## 10. Explainability & Interpretability
- **Global Attribution:** TreeSHAP reveals that discharge destination (`Home` vs `SNF/Facility`), historical utilization intensity, length of stay, prior inpatient hospitalizations, and circulatory comorbidities dominate prediction.
- **Local Attribution:** Patient waterfall decomposition separates factors that increase risk from factors that decrease risk.
- **Language Standard:** Phrased strictly as *"contributed to model output"* to prevent causal conflation.
- **Mathematical Counterfactuals:** Identifies minimum feature shifts required to move below operating threshold, explicitly labeled as non-medical sensitivity analyses.

---

## 11. Demographic Subgroup & Fairness Audit
Audited on held-out test data at operating threshold 0.125:
- **Age Strata:**
  - `< 50 years` (N=1,731, Base Rate=7.39%): Recall = 20.31%, Precision = 18.06%, F1 = 0.1912.
  - `50 - 69 years` (N=4,165, Base Rate=8.50%): Recall = 27.40%, Precision = 17.48%, F1 = 0.2134.
  - `70+ years` (N=4,603, Base Rate=10.02%): Recall = 44.90%, Precision = 15.71%, F1 = 0.2327.
- **Gender Strata:**
  - `Female` (N=5,666, Base Rate=9.11%): Recall = 38.37%, Precision = 16.56%, F1 = 0.2313.
  - `Male` (N=4,832, Base Rate=8.84%): Recall = 30.91%, Precision = 16.08%, F1 = 0.2115.
- **Race Strata:**
  - `African American` (N=1,890, Base Rate=9.15%): Recall = 36.99%, Precision = 20.32%, F1 = 0.2623.
  - `Caucasian` (N=7,820, Base Rate=8.99%): Recall = 35.14%, Precision = 15.49%, F1 = 0.2150.
  - `Other/Unreported` (N=789, Base Rate=8.49%): Recall = 28.36%, Precision = 17.76%, F1 = 0.2184.
- **Governance Statement:** Performance varies across demographic subgroups. Disparities stem from historical documentation patterns and epidemiological comorbidity differences in historical data.

---

## 12. Monitoring & Model Retraining
- Real-time statistical drift engine evaluates incoming batches against training baseline reference:
  - Population Stability Index (PSI) with thresholds: 🟢 Stable (<0.10), 🟡 Monitor (0.10–0.25), 🔴 Drift Detected (>=0.25).
  - Kolmogorov-Smirnov (KS) two-sample test for numerical distributional divergence.
- When >=3 features enter drift status, automated alerts recommend model review and parameter recalibration.

---

## 13. Limitations & Responsible AI
1. **Historical EHR Limitations:** Dataset reflects encounters from 1999–2008. Healthcare operations, medication regimens, and ICD standards (ICD-9 vs ICD-10) have evolved.
2. **Unobserved Confounders:** Social determinants of health (SDOH), income, food security, caregiver support, and outpatient pharmacy fill rates are absent in public hospital discharge datasets.
3. **Clinical Integration Protocol:** The platform must only be used as a triage prioritization filter for nurse care coordinators, never as an automated clinical decision agent.
