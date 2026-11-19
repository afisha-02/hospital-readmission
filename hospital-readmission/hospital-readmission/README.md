# Hospital Readmission Risk Prediction

Predicts whether a diabetic patient will be readmitted to hospital within 30 days, using the public UCI "Diabetes 130-US Hospitals" dataset (about 100,000 encounters). The focus is on doing it carefully: avoiding data leakage, picking a sensible decision threshold, calibrating the probabilities and explaining each prediction.

> Portfolio project on public data. It is not a clinical tool.

## Results

Evaluated on 10,499 held-out patients (8.98% were readmitted within 30 days):

| Model | PR-AUC | ROC-AUC | Recall | Precision | Brier |
| --- | :---: | :---: | :---: | :---: | :---: |
| Dummy (always "no readmission") | 0.090 | 0.500 | 0.000 | 0.000 | 0.082 |
| Logistic Regression | 0.163 | 0.638 | 0.522 | 0.134 | 0.232 |
| Random Forest | 0.157 | 0.634 | 0.384 | 0.152 | 0.207 |
| XGBoost, uncalibrated (0.50 cutoff) | 0.164 | 0.637 | 0.504 | 0.139 | 0.217 |
| XGBoost, calibrated (0.125 cutoff) | 0.164 | 0.637 | 0.350 | 0.164 | 0.080 |

What I take from this:

- Readmission is hard to predict from this data. The models beat chance but only modestly (ROC-AUC about 0.64).
- Accuracy is misleading. The dummy model is about 91% accurate and finds no readmissions, so I looked at PR-AUC, recall and precision.
- Calibration made the probabilities trustworthy: Brier score went from 0.217 to 0.080 and calibration error from 0.362 to 0.006.
- The threshold is a trade-off. Lowering it finds more readmissions but flags more patients who won't return, so I set it from a configurable cost matrix.

## How I avoided leakage

- Removed patients who died or went to hospice (they can't be readmitted)
- Kept only the first encounter per patient, so the same person never appears in both train and test
- Dropped `encounter_id`
- Fit imputers and scalers on the training set only, inside a scikit-learn pipeline

## What's in it

- Data validation and a leakage audit before any modelling
- Feature engineering (prior visits, diagnosis grouping, lab and medication summaries)
- Baselines, then XGBoost tuned with Optuna, with runs logged in MLflow
- Probability calibration and cost-based threshold selection
- SHAP explanations and a counterfactual "what if" tool
- Subgroup audit by age, sex and race
- Drift check on new data (PSI and KS tests)
- FastAPI service and a Streamlit dashboard
- 20 pytest tests

## Project structure

```
app/            FastAPI service (api.py) and Streamlit dashboard
src/            data, validation, features, preprocessing, models,
                calibration, thresholding, explainability, fairness, monitoring
configs/        config.yaml
notebooks/      six notebooks: data quality to evaluation
reports/        figures and JSON reports
models/         saved pipeline and model files
tests/
MODEL_CARD.md   intended use, limits and subgroup results
```

## Run it

```bash
git clone https://github.com/afisha-02/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

# data and training
python src/data/download_dataset.py
python src/validation/data_contract.py
python src/validation/leakage_audit.py
python src/preprocessing/pipeline.py
python src/models/baselines.py
python src/models/train_xgboost.py
python src/calibration/calibrator.py
python src/thresholding/optimizer.py
python src/explainability/shap_explainer.py
python src/fairness/subgroup_audit.py
python src/evaluation/generate_comparison.py

# tests
python -m pytest tests/

# dashboard and API
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload --port 8000
```

API docs are at `http://localhost:8000/docs`. Endpoints: `/health`, `/model-info`, `/predict`, `/predict/batch`, `/explain`, `/metrics`, `/drift`.

Docker: `docker-compose up --build`

## Limitations

- Public dataset from 1999 to 2008, from US hospitals only. Results may not carry over to other populations.
- Modest predictive power, and no prospective or clinical validation.
- The cost values behind the threshold are example numbers, not real hospital costs.
- Keeping only the first encounter per patient avoids leakage but ignores repeat-visit patterns.

## License

MIT. See [LICENSE](LICENSE).
