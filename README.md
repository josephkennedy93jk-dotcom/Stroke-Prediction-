# From Prediction to Prescription: A Stroke Risk Analytics Pipeline

**One-line pitch:** Most stroke projects stop at "predict who will have a stroke." This one asks the harder question — *for each patient, which intervention lowers their risk most, and by how much?*

A full analytics stack — **Descriptive → Predictive → Prescriptive** — built on the Kaggle Healthcare Stroke dataset (5,110 patients × 12 features, ~5% positive class).

---

## Dataset

- **Source:** [Kaggle Healthcare Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Rows:** 5,110 patients
- **Target:** `stroke` (binary — 1 = had a stroke, 0 = did not)
- **Class balance:** ~4.9% positive → heavily imbalanced (a real-world modelling problem, not a toy dataset)
- **Features:** `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

---

## 1. Clean

- **Duplicates:** 0 found.
- **Missing values:** 201 nulls in `bmi` (~4% of the column).
- **Imputation strategy:** Group-wise mean by `(gender, age)` — respects the fact that BMI varies systematically by both. Falls back to gender-only mean for any age × gender bucket that was fully missing (1 residual row).
- **`Other` gender row:** 1 row dropped from modelling — sample too small to learn from without leaking.
- **Encoding:**
  - Binary columns → 0/1 mapping (`gender`, `ever_married`, `Residence_type`)
  - Nominal columns → one-hot with `drop_first=True` (`work_type`, `smoking_status`)
- **Scaling:** `StandardScaler` for linear / Naive Bayes models. Trees left unscaled — they don't care.

---

## 2. Explore

The EDA phase profiles who ends up in the stroke cohort vs. the general population — the goal is to build intuition before touching a model.

Charts included in `Healthcare_dataset_exploratory_&_Descriptive.ipynb`:

- **Gender distribution** — dataset vs. stroke victims (pie charts, count + %).
- **Age distribution** — 5-year bins, KDE overlay, dataset vs. stroke victims side by side. This is where the age signal jumps out.
- **Youngest / oldest stroke victims** — sanity check on outliers (a 1.32-year-old shows up — real data is messy).
- **Categorical features** (`ever_married`, `work_type`, `Residence_type`, `smoking_status`) — count plots, dataset vs. stroke cohort.
- **Numerical features** (`avg_glucose_level`, `bmi`) — histograms with KDE.
- **Comorbidity totals** — total counts of hypertension, heart disease, and stroke across the sample.

---

## 3. Descriptive

Population-level takeaways surfaced by the EDA:

- **Age dominates.** Stroke incidence rises sharply after ~55 and heavily concentrates 65+.
- **Comorbidity matters.** Patients with hypertension *and* heart disease are massively over-represented in the stroke cohort relative to their base rate.
- **Glucose skews right.** Stroke victims show a long right tail on `avg_glucose_level` — the diabetic and pre-diabetic ranges are enriched.
- **Smoking status is not a clean signal.** "Unknown" is a large category and confounds interpretation.
- **Gender is roughly balanced** among stroke cases despite males being slightly under-represented in the dataset.

---

## 4. Predictive

Four base models + three ensembles, benchmarked head-to-head on a stratified 75/25 train/test split.

**Class imbalance handling:**
- Linear / RF / NB → `class_weight='balanced'` or equal priors
- XGBoost → `scale_pos_weight = neg / pos`
- Naive Bayes → `priors=[0.5, 0.5]`

### Models

| Model | Why it's here |
|---|---|
| **Logistic Regression** | Interpretable baseline — coefficients read like standardised factor loadings. |
| **XGBoost** | Gradient-boosted trees — nonlinear interactions + built-in feature importance. |
| **Random Forest** | Bagged trees with importance ± std across 500 trees. Also drives partial-dependence plots. |
| **Naive Bayes** | Fast probabilistic baseline. Its Gaussian assumption is deliberately exposed via distribution overlays. |
| **Voting (equal)** | Simple soft-vote average of the four base models. |
| **Voting (AUC-weighted)** | Performance-weighted average — heavier weight on stronger models. |
| **Stacking** | Meta-learner (logistic) trained on cross-validated base-model predictions. |

### Evaluation stack

For every model:

- **ROC curve + AUC**
- **Precision-Recall curve + Average Precision** (more informative than ROC under 5% base rate)
- **Confusion matrix**
- **Predicted probability distribution by true class** — visual separation quality
- **Calibration curve + Brier score** — how honest are the probabilities?

Model-specific charts:

- **Logistic:** standardised coefficient bar chart (red = raises risk, blue = lowers) + odds ratios.
- **XGBoost:** feature importance + full **SHAP summary** (dot + bar).
- **Random Forest:** feature importance with std bars across 500 trees + **partial dependence plots** for `age`, `avg_glucose_level`, `bmi`, `hypertension` (the functional-form chart).
- **Naive Bayes:** the model's *assumed* Gaussians overlaid on the actual per-feature distributions — literally shows where NB's worldview breaks.
- **Ensembles:** head-to-head ROC + PR overlay, leaderboard bar chart, calibration comparison.

---

## 5. Prescriptive

This is the layer that separates the project from a standard Kaggle notebook.

### Calibrated risk scoring

- Base model: logistic regression with `class_weight='balanced'` (chosen for interpretability + strong ranking).
- Wrapped in `CalibratedClassifierCV(method='sigmoid', cv=5)` — Platt scaling corrects the inflated probabilities from class weighting back to reality.
- Scores generated via `cross_val_predict` — every patient scored by a model that didn't see them during training.
- **Result:** predicted `stroke_risk_%` where a 15% risk actually corresponds to ~15% observed stroke rate in that decile.

### Risk banding

| Band | Threshold | Colour |
|---|---|---|
| Green (Low) | < 2% | 🟢 |
| Yellow (Moderate) | 2 – 5% | 🟡 |
| Orange (High) | 5 – 15% | 🟠 |
| Red (Very High) | > 15% | 🔴 |

### Counterfactual interventions

For each patient the pipeline runs a set of **what-if scenarios** through the fitted model:

- What if hypertension were controlled (`hypertension = 0`)?
- What if heart disease were aggressively managed (`heart_disease = 0`)?
- What if glucose were reduced to normal (~100 mg/dL)?
- What if BMI were reduced to healthy range (~25)?
- What if the patient quit smoking?
- **Combined:** all modifiable factors addressed at once.

Each scenario returns the new risk %, the absolute change (pp), and the relative reduction (%).

### Per-patient outputs

- Individual **SHAP waterfall charts** (P&L attribution-style — red bars raise risk, blue lower) for representative patients across Green / Yellow / Orange / Red bands, including a true-positive and a false-positive in the Red band.
- A natural-language `prescriptive_advice` string attached to every row: risk profile + ranked recommended actions + combined-intervention outcome.

### Deliverables

- `stroke_data_with_prescriptive_advice.csv` — full scored dataset with risk % + band + advice text.
- `stroke_data_with_prescriptive_advice.xlsx` — same, with conditional formatting on the band column and a 3-colour scale on the risk % column.
- Population-level chart: number of patients addressable by each intervention.

---

## Limitations

- **Screening tool, not a diagnostic.** With a ~5% base rate, high recall comes at the cost of precision at every threshold. This model flags candidates for follow-up; it does not diagnose.
- **Observational data.** Counterfactuals ("if hypertension were controlled") are *model-implied* risk reductions, not causal estimates. A patient with hypertension differs from a patient without it in many unobserved ways.
- **Feature set is limited.** No cholesterol, no blood pressure numeric, no medication history, no imaging. A production model would look very different.
- **Smoking `Unknown` is a large category** and treated as its own level rather than imputed — safer, but reduces the smoking signal.
- **Single dataset, no external validation.** Results should be re-checked on a held-out cohort from a different source before any real-world use.
- **Age-driven predictions.** Age dominates — much of the "high risk" flagging is really "old." The counterfactual layer helps because it isolates the *modifiable* portion of risk.
- **One `Other` gender row was dropped** for modelling. Not ideal from a fairness standpoint but unavoidable with n=1.

---

## Tools used

**Language & environment**
- Python 3 (Google Colab)
- Jupyter notebooks

**Core data**
- `pandas` — data wrangling
- `numpy` — numerics

**Modelling**
- `scikit-learn` — logistic regression, random forest, naive bayes, `StandardScaler`, `train_test_split`, `CalibratedClassifierCV`, `cross_val_predict`, `VotingClassifier`, `StackingClassifier`, metrics, `PartialDependenceDisplay`
- `xgboost` — gradient boosting
- `shap` — model explainability (linear + tree explainers)

**Visualisation**
- `matplotlib`
- `seaborn`

**Output / export**
- `xlsxwriter` — styled Excel export with conditional formatting

---

## Repo structure

```
stroke prediction project/
├── README.md                                             ← this file
├── healthcare-dataset-stroke-data.csv                    ← raw input
├── Healthcare_dataset_exploratory_&_Descriptive.ipynb    ← 1. Clean + 2. Explore + 3. Descriptive
├── Healthcare_dataset_predictive_analytics.ipynb         ← 4. Predictive (4 models + 3 ensembles)
├── Copy_of_Healthcare_dataset_prescriptive_analytics.ipynb ← 5. Prescriptive (calibrated + counterfactual)
├── stroke_data_with_prescriptive_advice.csv              ← final scored output
└── stroke_data_with_prescriptive_advice.xlsx             ← styled Excel version
```

---

## How to run

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn xlsxwriter
   ```
3. Open the notebooks in order:
   - `Healthcare_dataset_exploratory_&_Descriptive.ipynb`
   - `Healthcare_dataset_predictive_analytics.ipynb`
   - `Copy_of_Healthcare_dataset_prescriptive_analytics.ipynb`
4. Update the file paths in the first cells if you're running locally instead of Colab.

---

