# Project proposal (as submitted)

Transcribed from `ECRI_Reading_Performance_Project_Proposal.docx` so the repo
carries its own statement of intent. Where later work contradicts the proposal,
`docs/findings.md` wins and says so explicitly.

## 1. Problem

Early reading ability is a foundation for academic development, but students are
often identified as needing support only after they have fallen behind, when
intervention is less effective. We build a model that predicts future reading
performance from earlier assessment scores, student-level information,
classroom variables and intervention features — a decision-support tool for
spotting students who would benefit from earlier, more targeted support, not a
replacement for teachers or reading specialists.

## 2. Dataset

The Enhanced Core Reading Instruction (ECRI) dataset from the NeurIPS 2024
Datasets & Benchmarks paper *Scalable Early Childhood Reading Performance
Prediction* — 44 schools, 6,916 students, 172 teachers. Longitudinal and
tabular: repeated measurements per student, so earlier information can predict
a later outcome. Contains demographics, reading assessments, intervention /
control indicators and classroom or teacher variables.

Planned preprocessing: clean column names, check duplicates, handle missing
values, encode categoricals, standardise numerics, and keep input features
strictly separate from target outcomes — the headline risk on longitudinal
education data is using future information to predict a future outcome.

## 3. Initial approach

Study the paper, then EDA: sample counts, feature types, missingness patterns,
score distributions, and the relationship between early reading measures and
later outcomes. Start with simple baselines to establish task difficulty —
logistic/linear regression, random forest, XGBoost or LightGBM, and a small
MLP — before anything more advanced. Classification metrics (accuracy,
precision, recall, F1, ROC-AUC, confusion matrix) or regression metrics (RMSE,
MAE, R²) depending on the final task definition, plus feature-importance
analysis so the model is interpretable.

Optional: a Streamlit "Early Reading Risk Predictor" app that returns a
predicted risk level with the factors driving it.

## 4. Expected challenges

- **Missing data** — longitudinal coverage is uneven; compare imputation
  strategies and prefer models robust to missingness.
- **Class imbalance** — if at-risk students are a small minority, use class
  weights, resampling or threshold tuning to protect recall.
- **Data leakage** — the biggest technical risk; define the prediction timeline
  carefully and keep future outcomes out of the inputs.
- **Interpretability** — education predictions must be explainable; use feature
  importance, SHAP-style explanations and clear visualisations.

## 5. Team roles

| Role | Main responsibilities |
|---|---|
| Paper and dataset review | Read the paper, document the variables, confirm the prediction target. |
| Data preprocessing | Clean, handle missing values, encode features, build train/test splits. |
| Modeling | Train baselines and compare approaches. |
| Evaluation and interpretation | Metrics, plots, feature-importance explanation. |
| Report and app development | Final report, and the Streamlit demo if time permits. |

## 6. Expected outcome

A working ML pipeline predicting early childhood reading performance or risk
level from ECRI, with data analysis, model comparison, evaluation results and
feature interpretation — plus the Streamlit app if time allows.

## References

1. Shangguan, Huang, Ohn-Bar, Ozernov-Palchik, Kosty, Stoolmiller, Fien.
   "Scalable Early Childhood Reading Performance Prediction." NeurIPS 2024
   Datasets & Benchmarks Track.
   <https://proceedings.neurips.cc/paper_files/paper/2024/hash/862170d83a17b7c2ef3c5a58e7f1992d-Abstract-Datasets_and_Benchmarks_Track.html>
2. Project page and data/code links: <https://ecri-data.github.io/>
