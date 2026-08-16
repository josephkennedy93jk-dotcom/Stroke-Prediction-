<h1 align="center">Stroke Risk Analytics — From Prediction to Prescription</h1>

<p align="center">
  <em>A full analytics stack — Descriptive → Predictive → Prescriptive — asking not just <strong>who</strong> is at risk, but <strong>what to do about it</strong>.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square"/>
  <img src="https://img.shields.io/badge/scikit--learn-Modelling-F7931E?style=flat-square"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB6E4B?style=flat-square"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP-6C4CB4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Calibration-Platt%20Scaling-4479A1?style=flat-square"/>
  <img src="https://img.shields.io/badge/Method-Counterfactual-brightgreen?style=flat-square"/>
</p>

---

## 1. About This Project

Most stroke-prediction projects stop at *"predict who will have a stroke."* This one asks the harder, clinically useful question: **for each patient, which intervention lowers their risk most, and by how much?**

Please read the accompanying presentation for the project: | Project presentation | [Stroke Predictive Analysis.pdf](Project%20Presentation/Stroke%20Predictive%20Analysis.pdf) |

Built on the Kaggle Healthcare Stroke dataset (5,110 patients × 12 features, ~5% positive class), the pipeline delivers a full analytics stack end-to-end:

- Cleaned the dataset with a group-wise BMI imputation strategy (not a naive column mean) and honest handling of edge cases
- Profiled the population — dataset vs. stroke cohort — to build intuition before touching a model
- Benchmarked four base models plus three ensembles on a stratified split, with class imbalance handled model by model
- Calibrated the winning model with Platt scaling so the risk percentages actually mean what they say
- Produced per-patient counterfactuals — "if hypertension were controlled, this patient's risk drops from X% to Y%" — and translated them into a natural-language advice string

---

## 2. The Business Question

Stroke is the second-leading cause of death globally and a leading cause of adult disability. Screening programmes need to decide who to call in, and clinicians need to know which lever to pull for each patient. A binary "will they / won't they" prediction is not enough to support either decision.

The analytics stack was designed to answer four questions:

- **Who is at elevated risk?** — a per-patient probability that's actually calibrated, not just a ranking.
- **Why are they at risk?** — feature-level attribution so the clinician can see the drivers, not just the score.
- **What's the modifiable portion of their risk?** — separating age-driven risk (fixed) from lifestyle/comorbidity risk (actionable).
- **Which intervention should be prioritised for this patient?** — a ranked, patient-specific action list, not a generic guideline.

Everything that follows is organised around answering those four questions in order.

---

## 3. Executive Summary

Age dominates stroke risk in the dataset, but the modifiable portion — hypertension, heart disease, glucose, BMI, smoking — is where the analytics stack adds value. Calibrated per-patient risk scores plus counterfactual "what if" modelling produce a ranked intervention list for every patient in the sample, banded from Green (Low) to Red (Very High).

The stack shifts the framing of the problem from *"predict stroke"* to *"prescribe the intervention that lowers this patient's risk the most" — a screening and triage tool clinicians can act on.

<p align="center">
  <img src="Project%20Images/stroke%20victim%20distribution.png" alt="Stroke Victim Distribution" width="720">
</p>

| Metric | Value |
|---|---|
| Patients in analytical base | 5,110 |
| Baseline stroke rate | ~4.9% (heavily imbalanced) |
| Base models benchmarked | 4 (Logistic · XGBoost · Random Forest · Naive Bayes) |
| Ensembles benchmarked | 3 (Voting equal · Voting AUC-weighted · Stacking) |
| Prescriptive layer | Calibrated risk % + risk band + counterfactual advice per patient |

---

## 4. Key Findings

The analysis surfaced five findings that shaped the prescriptive layer.

- **Age dominates — but that's not the whole story.** Stroke incidence climbs sharply after ~55 and heavily concentrates 65+. Any model that scores patients by risk is largely scoring by age unless the modifiable component is isolated.
- **Comorbidity is a force multiplier.** Patients with hypertension *and* heart disease are massively over-represented in the stroke cohort relative to their base rate. These are the highest-leverage intervention targets.
- **Glucose has a long right tail.** Stroke victims show elevated `avg_glucose_level` — the diabetic and pre-diabetic ranges are visibly enriched, consistent with the metabolic-vascular pathway.
- **Smoking status is a noisy signal.** The `Unknown` category is large enough to confound interpretation; the signal is real but weaker than the population narrative suggests.
- **Probabilities need calibration, not just ranking.** Class weighting inflates raw probabilities. Wrapping the model in `CalibratedClassifierCV` (Platt scaling) is what makes a "15% risk" mean 15% observed stroke rate in that decile — the requirement for a prescriptive layer.

---

## 5. Recommendations to Clinical & Screening Programme Leadership

The following recommendations are drawn from the findings above and are intended as strategic areas to explore for a screening or preventative programme built on this style of analytics.

### 1. Frame the model as a screening tool, not a diagnostic
At a ~5% base rate, high recall inevitably means low precision — this model flags candidates for follow-up, not people to treat. Communicate that framing explicitly to clinicians and patients from day one.

### 2. Prioritise the modifiable-risk view over the raw risk score
Because age dominates, raw risk scores concentrate on the elderly by default. The counterfactual layer (see Section 10) isolates the actionable portion. That view should be the front page of any clinician-facing dashboard, not an appendix.

### 3. Target hypertension and heart disease as the two highest-leverage levers
Both features are massively over-represented in the stroke cohort. Standard-of-care management of these two conditions is where the counterfactual analysis suggests the greatest per-patient risk reduction can be achieved.

### 4. Treat glucose management as the third pillar
The right-tail concentration on `avg_glucose_level` supports integrating diabetic and pre-diabetic screening with the stroke-risk workflow rather than running them as separate programmes.

### 5. Insist on calibrated probabilities before any clinical use
An uncalibrated model that ranks well but reports "80% risk" for a patient with a true 20% risk is dangerous in a screening context. Platt scaling (or isotonic regression) is a non-negotiable step, not an optional one.

### 6. Communicate risk in bands, not numbers
Patients and non-specialist staff read Green / Yellow / Orange / Red faster than they read percentages. The banding structure (see Section 11) is designed for exactly that.

### 7. Validate externally before deployment
Every result here comes from a single dataset. Before any real-world use, results should be re-checked on a held-out cohort from a different source, with fairness audits across gender and age groups.

### Areas to explore further
Deeper investigation is warranted in three areas: adding richer clinical features (cholesterol, blood pressure numeric, medication history), formalising the counterfactuals as causal estimates rather than model-implied deltas, and testing the intervention rankings against clinician judgement on a labelled sample.

---

## 6. Data Foundation

The analysis draws on the Kaggle Healthcare Stroke dataset — one row per patient, covering demographics, clinical comorbidities, lifestyle factors and stroke outcome.

**Scope**
- 5,110 patient records
- Target: `stroke` (binary — 1 = had a stroke, 0 = did not)
- Class balance: ~4.9% positive — a genuinely imbalanced real-world problem
- Features: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

**Source:** [Kaggle Healthcare Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Raw source data: [healthcare-dataset-stroke-data.csv](Raw%20CSV%20Data/healthcare-dataset-stroke-data.csv)

<p align="center">
  <img src="Project%20Images/distribution%20of%20columns.png" alt="Distribution of columns across the dataset" width="820">
</p>

---

## 7. Data Cleaning & Preparation

Data-quality treatment was deliberate rather than defaulted — the imputation strategy matters more than the model for a dataset this size.

- **Duplicates:** 0 found.
- **Missing values:** 201 nulls in `bmi` (~4% of the column).
- **Imputation strategy:** group-wise mean by `(gender, age)` — respects the fact that BMI varies systematically by both. Falls back to gender-only mean for any age × gender bucket that was fully missing (1 residual row). Chosen over a naive column mean because it preserves the distributional structure the model needs to see.
- **`Other` gender row:** 1 row dropped from modelling — sample too small to learn from without leaking.
- **Encoding:**
  - Binary columns → 0/1 mapping (`gender`, `ever_married`, `Residence_type`)
  - Nominal columns → one-hot with `drop_first=True` (`work_type`, `smoking_status`)
- **Scaling:** `StandardScaler` for linear and Naive Bayes models. Trees left unscaled — they don't care.

---

## 8. Exploratory & Descriptive Analysis

The exploratory phase profiles who ends up in the stroke cohort vs. the general population — the goal is to build intuition before touching a model.

<p align="center">
  <img src="Project%20Images/male%20female%20pie%20charts.png" alt="Gender distribution — dataset vs. stroke cohort" width="720">
</p>

<p align="center">
  <img src="Project%20Images/stroke%20to%20age%20etc.png" alt="Age distribution — stroke vs. general" width="820">
</p>

<p align="center">
  <img src="Project%20Images/glucose%20and%20bmi.png" alt="Glucose and BMI distributions" width="820">
</p>

<p align="center">
  <img src="Project%20Images/health%20conditions%20in%20sample.png" alt="Comorbidity totals across the sample" width="720">
</p>

<p align="center">
  <img src="Project%20Images/more%20distributions.png" alt="Categorical feature distributions" width="820">
</p>

**Population-level takeaways**

- **Age dominates.** Stroke incidence rises sharply after ~55 and heavily concentrates 65+.
- **Comorbidity matters.** Patients with hypertension *and* heart disease are massively over-represented in the stroke cohort relative to their base rate.
- **Glucose skews right.** Stroke victims show a long right tail on `avg_glucose_level` — the diabetic and pre-diabetic ranges are enriched.
- **Smoking status is not a clean signal.** "Unknown" is a large category and confounds interpretation.
- **Gender is roughly balanced** among stroke cases despite males being slightly under-represented in the dataset.

Notebook: [Healthcare_dataset_exploratory_&_Descriptive.ipynb](Python%20Files/Healthcare_dataset_exploratory_%26_Descriptive.ipynb)

---

## 9. Predictive Modelling & Performance

Four base models and three ensembles, benchmarked head-to-head on a stratified 75/25 train/test split. Class imbalance is handled per model — `class_weight='balanced'` for linear / RF, `scale_pos_weight = neg / pos` for XGBoost, `priors=[0.5, 0.5]` for Naive Bayes.

| Model | Why it's here |
|---|---|
| **Logistic Regression** | Interpretable baseline — coefficients read like standardised factor loadings. |
| **XGBoost** | Gradient-boosted trees — nonlinear interactions + built-in feature importance. |
| **Random Forest** | Bagged trees with importance ± std across 500 trees. Also drives partial-dependence plots. |
| **Naive Bayes** | Fast probabilistic baseline. Its Gaussian assumption is deliberately exposed via distribution overlays. |
| **Voting (equal)** | Simple soft-vote average of the four base models. |
| **Voting (AUC-weighted)** | Performance-weighted average — heavier weight on stronger models. |
| **Stacking** | Meta-learner (logistic) trained on cross-validated base-model predictions. |

**Evaluation stack applied to every model:** ROC + AUC, Precision-Recall + Average Precision (more informative than ROC under 5% base rate), confusion matrix, predicted-probability distribution by true class, calibration curve + Brier score.

**Logistic Regression**

<p align="center">
  <img src="Project%20Images/logistic%20metrics.png" alt="Logistic regression metrics" width="820">
</p>

<p align="center">
  <img src="Project%20Images/logistic%20regression%20factors.png" alt="Logistic regression standardised coefficients" width="820">
</p>

**XGBoost**

<p align="center">
  <img src="Project%20Images/xg%20boost%20metrics.png" alt="XGBoost metrics" width="820">
</p>

<p align="center">
  <img src="Project%20Images/xg%20boost%20features%20imp.png" alt="XGBoost feature importance + SHAP" width="820">
</p>

**Random Forest**

<p align="center">
  <img src="Project%20Images/random%20forrest%20metrics.png" alt="Random Forest metrics" width="820">
</p>

**Naive Bayes**

<p align="center">
  <img src="Project%20Images/naive%20bayes%20performance.png" alt="Naive Bayes performance with Gaussian overlay" width="820">
</p>

**Ensembles**

<p align="center">
  <img src="Project%20Images/ensemble%20scores%20predict.png" alt="Ensemble ROC + PR overlay and leaderboard" width="820">
</p>

Notebook: [Healthcare_dataset_predictive_analytics.ipynb](Python%20Files/Healthcare_dataset_predictive_analytics.ipynb)

---

## 10. Prescriptive Layer — Calibrated Risk & Counterfactuals

This is the layer that separates the project from a standard Kaggle notebook.

**Calibrated risk scoring**

- Base model: logistic regression with `class_weight='balanced'` (chosen for interpretability and strong ranking).
- Wrapped in `CalibratedClassifierCV(method='sigmoid', cv=5)` — Platt scaling corrects the inflated probabilities from class weighting back to reality.
- Scores generated via `cross_val_predict` — every patient scored by a model that didn't see them during training.
- **Result:** predicted `stroke_risk_%` where a 15% risk actually corresponds to ~15% observed stroke rate in that decile.

<p align="center">
  <img src="Project%20Images/prescriptive%20predicted%20stroke%20risk.png" alt="Prescriptive predicted stroke risk" width="820">
</p>

<p align="center">
  <img src="Project%20Images/predicted%20vs%20actual%20stroke%20risk%20prescriptive.png" alt="Predicted vs actual stroke risk" width="820">
</p>

**Counterfactual interventions**

For each patient the pipeline runs a set of **what-if scenarios** through the fitted model:

- What if hypertension were controlled (`hypertension = 0`)?
- What if heart disease were aggressively managed (`heart_disease = 0`)?
- What if glucose were reduced to normal (~100 mg/dL)?
- What if BMI were reduced to healthy range (~25)?
- What if the patient quit smoking?
- **Combined:** all modifiable factors addressed at once.

Each scenario returns the new risk %, the absolute change (pp), and the relative reduction (%).

**Per-patient outputs**

- Individual **SHAP waterfall charts** (P&L attribution-style — red bars raise risk, blue lower) for representative patients across Green / Yellow / Orange / Red bands, including a true-positive and a false-positive in the Red band.
- A natural-language `prescriptive_advice` string attached to every row: risk profile + ranked recommended actions + combined-intervention outcome.

Notebook: [Copy_of_Healthcare_dataset_prescriptive_analytics.ipynb](Python%20Files/Copy_of_Healthcare_dataset_prescriptive_analytics.ipynb)

---

## 11. Risk Banding & Patient Communication

Every patient is placed into a risk band based on their calibrated stroke probability. Bands drive both the clinical playbook and the patient-facing communication tone.

| Band | Threshold | Programme role |
|---|---|---|
| Green (Low) | < 2% | Monitor at routine cadence |
| Yellow (Moderate) | 2 – 5% | Lifestyle counselling + annual review |
| Orange (High) | 5 – 15% | Active management of comorbidities + 6-month review |
| Red (Very High) | > 15% | Priority clinical review + counterfactual-guided intervention plan |

<p align="center">
  <img src="Project%20Images/stroke%20risk%20distribution%20prescriptive.png" alt="Stroke risk distribution across bands" width="820">
</p>

<p align="center">
  <img src="Project%20Images/red%20band%20patient%20breakdown.png" alt="Red-band patient breakdown" width="820">
</p>

**Patient-facing risk alert concept**

<p align="center">
  <img src="Project%20Images/Pateint%20app%20stroke%20risk%20alert%20.png" alt="Patient app stroke risk alert concept" width="720">
</p>

A patient-facing alert format built on the same banding — the point being that a calibrated probability plus a ranked action list can be delivered to the patient directly, not just the clinician.

---

## 12. Recommended Clinical Playbook

| Band | Treatment | Expected effect |
|---|---|---|
| **Red (Very High)** | Priority clinician review · counterfactual-guided intervention plan (hypertension, heart disease, glucose, BMI, smoking, in ranked order per patient) · 30-day follow-up | Address the highest-leverage modifiable levers first for the patients with the most room to move |
| **Orange (High)** | Active management of comorbidities · lifestyle counselling · six-month review · re-score on next data refresh | Efficient, scalable intervention on the mid-risk segment where the calibrated probability is meaningfully above baseline |
| **Yellow (Moderate)** | Lifestyle counselling · annual review · patient-facing app alert with modifiable-risk education | Prevent tier-drift into Orange; maintain engagement |
| **Green (Low)** | Routine monitoring · population-level messaging | Standard care with no additional cost |

---

## 13. Limitations

Analytical honesty — the failure modes worth naming explicitly.

- **Screening tool, not a diagnostic.** With a ~5% base rate, high recall comes at the cost of precision at every threshold. This model flags candidates for follow-up; it does not diagnose.
- **Observational data.** Counterfactuals ("if hypertension were controlled") are *model-implied* risk reductions, not causal estimates. A patient with hypertension differs from a patient without it in many unobserved ways.
- **Feature set is limited.** No cholesterol, no blood pressure numeric, no medication history, no imaging. A production model would look very different.
- **Smoking `Unknown` is a large category** and treated as its own level rather than imputed — safer, but reduces the smoking signal.
- **Single dataset, no external validation.** Results should be re-checked on a held-out cohort from a different source before any real-world use.
- **Age-driven predictions.** Age dominates — much of the "high risk" flagging is really "old." The counterfactual layer helps because it isolates the *modifiable* portion of risk.
- **One `Other` gender row was dropped** for modelling. Not ideal from a fairness standpoint but unavoidable with n=1.

---

## 14. Deliverables

The tangible outputs of the engagement, each linked below:

| Deliverable | File |
|---|---|
| Project presentation | [Stroke Predictive Analysis.pdf](Project%20Presentation/Stroke%20Predictive%20Analysis.pdf) |
| Exploratory & descriptive notebook | [Healthcare_dataset_exploratory_&_Descriptive.ipynb](Python%20Files/Healthcare_dataset_exploratory_%26_Descriptive.ipynb) |
| Predictive modelling notebook | [Healthcare_dataset_predictive_analytics.ipynb](Python%20Files/Healthcare_dataset_predictive_analytics.ipynb) |
| Prescriptive analytics notebook | [Copy_of_Healthcare_dataset_prescriptive_analytics.ipynb](Python%20Files/Copy_of_Healthcare_dataset_prescriptive_analytics.ipynb) |
| Scored patient file (CSV) | [stroke_data_with_prescriptive_advice.csv](Raw%20CSV%20Data/stroke_data_with_prescriptive_advice.csv) |
| Scored patient file (styled Excel) | [stroke_data_with_prescriptive_advice.xlsx](Raw%20CSV%20Data/stroke_data_with_prescriptive_advice.xlsx) |
| Raw source data | [healthcare-dataset-stroke-data.csv](Raw%20CSV%20Data/healthcare-dataset-stroke-data.csv) |

---

## 15. What I'd Do Next

Given more time or a follow-up engagement, five extensions would materially strengthen the stack:

- **Add richer clinical features** — cholesterol, blood pressure numeric, medication history, imaging findings. The current feature set is the ceiling on how much this model can improve.
- **Move counterfactuals from model-implied to causal** — propensity-score matching or a double-machine-learning approach would let intervention effects be estimated rather than assumed.
- **External validation on a different cohort** — the single-dataset caveat is the biggest barrier to real-world use.
- **Fairness audit** — subgroup performance across gender, age band, and work type before any deployment.
- **Serve the model behind a clinician-facing tool** — a lightweight app that takes patient input, returns the calibrated risk, the band, the SHAP waterfall and the ranked intervention list in one view. The patient-alert mockup in Section 11 is the starting point.

---

## 16. Author

**Joseph Kennedy** — Data Analyst

End-to-end delivery: data cleaning and imputation, exploratory analysis, predictive modelling (logistic, XGBoost, random forest, naive bayes, voting and stacking ensembles), probability calibration, per-patient counterfactuals, SHAP explainability, and stakeholder communication.

<sub>Built on the publicly available Kaggle Healthcare Stroke dataset (5,110 patients). Results have not been externally validated and are for analytical demonstration only — not clinical use.</sub>

