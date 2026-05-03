# From Neurology to Code: Stroke Risk Prediction with Calibrated Probability Models

## Important: please read before using

This is an educational machine learning exercise on a public Kaggle dataset. It is not medical advice and not a clinical decision-support tool. The model produced here has not been validated for clinical use, has not been reviewed by any regulatory body, and must not be used to diagnose, screen, treat, or make any decision about real patients.

I am working on this as a software and machine learning engineer, not as a treating physician. The clinical commentary in the notebook exists to show how clinical reasoning can inform ML feature engineering and model evaluation. It is not the practice of medicine.

If you are seeking medical evaluation, talk to a licensed clinician in your jurisdiction. If you are building a real clinical tool, follow the FDA SaMD pathway, ISO 14155, and IEC 62304 — and validate prospectively before any patient ever sees the output.

## Overview

This is the second notebook in my "From Neurology to Code" series, where I work through machine learning problems with the same diagnostic discipline I once brought to clinical reasoning. The notebook builds a calibrated, audited, deployment-packaged stroke risk prediction model on a public dataset of 5,110 patients.

[View the full interactive notebook on Kaggle](https://www.kaggle.com/code/alirezaminagar/from-neurology-to-code-stroke-risk-prediction-wit)

The notebook is hosted on Kaggle as the single source of truth — runnable in-browser, with all outputs and visualizations rendered live.

## What this notebook does that most don't

Most Kaggle medical notebooks stop at ROC-AUC. This one goes further:

- Calibration analysis — measures the probability distortion with reliability diagrams and Brier score, then corrects it with isotonic regression
- Threshold selection — explicit sensitivity-specificity tradeoff with precision-recall curves and clinical reasoning, instead of accepting the default 0.5
- SHAP interpretability — global feature importance with a clinical sanity check on the top features
- Fairness audit — calibration and recall stratified by sex and age group, surfacing a structural failure that aggregate metrics hid
- Deployment packaging — saved sklearn pipeline plus a callable inference function with metadata and built-in scope warnings

These five sections are what set the notebook apart from a typical Kaggle medical entry. They reflect the engineering standards a real clinical AI system would have to meet.

## Methods

| Component | Approach |
|---|---|
| Data | Kaggle Stroke Prediction Dataset — 5,110 patients, 12 features, 4.87% positive class |
| Preprocessing | sklearn Pipeline with ColumnTransformer for leak-proof cross-validation |
| Feature engineering | Clinical bins for age, BMI, and glucose; comorbidity counts; age-by-hypertension interaction; pediatric flag |
| Models compared | Logistic regression baseline plus LightGBM, XGBoost, and CatBoost |
| Validation | 5-fold stratified cross-validation with a locked test set, touched once at the end |
| Imbalance handling | Class weights and scale_pos_weight matched to the 19.5-to-1 negative-to-positive ratio |
| Calibration | Isotonic regression via CalibratedClassifierCV with internal 5-fold CV |
| Threshold selection | Precision-recall curves, F1 optimization, and sensitivity-specificity-precision tradeoff plots |
| Interpretability | SHAP TreeExplainer for global feature importance |
| Fairness audit | Subgroup recall, precision, and Brier score by sex and age group |
| Deployment | Joblib-serialized pipeline with a predict_stroke_risk function and scope warnings |

## Key findings

| Metric | Value | Interpretation |
|---|---|---|
| ROC-AUC (test set) | 0.82 | Strong ranking ability |
| PR-AUC (test set) | 0.17 | About 4 times better than the 0.049 no-skill baseline |
| Brier score (calibrated) | 0.043 | Below the no-skill floor of 0.046 |
| Sensitivity at threshold 0.05 | 0.78 overall, 0.91 older adult, 1.00 elderly | High-recall screening operating point |
| Specificity at threshold 0.05 | 0.69 | Reasonable for a triage tool |
| Sex fairness | Equal recall and Brier across male and female | No detected disparity |
| Age fairness | Catastrophic failure under age 45 | Deployment scope restricted to adults aged 45 and above |

### The headline lesson

The age-fairness audit is the most important finding in the notebook. The aggregate ROC-AUC of 0.82 looked acceptable. Stratifying by age group revealed that the model misses essentially every stroke in patients under 45. The model has learned that strokes don't happen below middle age, and no other clinical feature in the dataset can override that prior. This is invisible without explicit subgroup analysis, and it is the failure mode that destroys real medical AI products in deployment.

The fix is not a better model. It is scope restriction — deploy only for adults aged 45 and above — and, if pediatric coverage is needed, a separate pediatric-specific model trained on pediatric-specific features like sickle cell status, congenital cardiac anomalies, and vascular malformations, which this dataset does not contain.

## Repository contents

This repository contains the README and the LICENSE. The notebook itself lives on Kaggle as the single source of truth.

## How to reproduce

The notebook lives on Kaggle and runs in-browser without local setup. The fastest way to reproduce:

1. Visit the [Kaggle notebook](https://www.kaggle.com/code/alirezaminagar/from-neurology-to-code-stroke-risk-prediction-wit)
2. Click "Copy & Edit" to fork it into your own Kaggle account
3. Click "Run All" to execute every cell

If you prefer to run it locally, use Kaggle's "Download Notebook" feature to get the .ipynb file, then install pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, shap, matplotlib, seaborn, and joblib. You'll also need to download the Stroke Prediction Dataset from Kaggle and update the file path in the data-loading cell.

## Tech stack

Python 3.12, pandas, numpy, scikit-learn (Pipeline, ColumnTransformer, CalibratedClassifierCV, StratifiedKFold, metrics), LightGBM, XGBoost, CatBoost, SHAP (TreeExplainer), matplotlib, seaborn, joblib.

## "From Neurology to Code" series

| Notebook | Domain | Technique |
|---|---|---|
| 1 — [Irrigation Prediction with LightGBM](https://www.kaggle.com/code/alirezaminagar/from-neurology-to-code-irrigation-prediction-with) | Tabular, agriculture | LightGBM with 5-fold CV |
| 2 — [Stroke Risk Prediction with Calibrated Probability Models](https://www.kaggle.com/code/alirezaminagar/from-neurology-to-code-stroke-risk-prediction-wit) — this repository | Tabular, clinical | Calibrated LightGBM with fairness audit |
| 3 — Adverse Drug Event Detection with DistilBERT — forthcoming | NLP, pharmacovigilance | DistilBERT fine-tuning |

The series documents the methodology of moving from clinical reasoning to applied ML engineering — same diagnostic discipline, different domain.

## About the author

## About the author

Alireza Minagar — Software and Machine Learning Engineer building healthcare AI portfolios. Adjunct Professor at the University of Maryland Global Campus, where I teach graduate-level software engineering and AI-augmented development. Background in academic neurology with a substantial publication record in computational neuroscience and neuro-immunology. AWS Certified Machine Learning – Specialty.

- LinkedIn: [linkedin.com/in/alireza-minagar-b450aa173](https://www.linkedin.com/in/alireza-minagar-b450aa173)
- GitHub: [github.com/aliminagar](https://github.com/aliminagar)
- Kaggle: [kaggle.com/alirezaminagar](https://www.kaggle.com/alirezaminagar)

## License

Released under the MIT License — see the LICENSE file for details. The dataset is published by fedesoriano on Kaggle and is synthetic and publicly available.

## Citation

If this work is useful in your own research or learning, please cite as:

Minagar, A. (2026). From Neurology to Code: Stroke Risk Prediction with Calibrated Probability Models. GitHub repository. https://github.com/aliminagar/from-neurology-to-code-stroke-prediction

---

If this repository was useful, please star it. Comments, corrections, and pull requests are welcome.
