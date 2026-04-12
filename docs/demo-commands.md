# Demo - ML Classification Pipeline

## 1. Clone the Repository

```bash
git clone https://github.com/lararli/ml-classificacao-b2.git
cd ml-classificacao-b2
```

## 2. Setup Environment

Creates a virtual environment and installs all dependencies.

```bash
make setup
```

## 3. Activate Virtual Environment

```bash
source .venv/bin/activate
```

## 4. Validate Configs

Loads all configuration files (data, pipeline, quality, experiments) and checks they parse correctly in both production and experimentation modes. Acts as a gate before running anything.

```bash
make configs
```

---

# Phase 1: Experimentation

## 5. Run Experimentation Pipeline

Runs the full pipeline in experimentation mode — 6 jobs in sequence:

1. **Ingestion** — downloads the dataset, converts to Parquet, validates schema
2. **Quality** — runs null checks, duplicate checks, saves quality report
3. **Train** — splits data, preprocesses, trains all models from config, logs metrics to MLflow
4. **Selection** — compares models by f1_score, runs financial impact analysis, picks the best
5. **Inference** — runs a test prediction with the best model to validate the pipeline
6. **Drift** — checks for statistical drift and new categories in features

```bash
make test
```

## 6. Compare Experimentation Results

Queries MLflow and displays a table with all models sorted by f1_score. Shows accuracy, precision, recall, gap (overfitting indicator), training time, and AUC-ROC.

```bash
make compare ENV=test
```

## 7. Analyze Experimentation

Full analysis: data samples, KPIs in percentages, approved vs rejected profiles, feature importances from the best model, and per-sample reasoning showing which factors drove each decision.

```bash
make analyze ENV=test
```

## 8. MLflow UI (Experimentation)

Open the MLflow dashboard to visually explore experimentation runs.

```bash
make mlflow
```

Access at: http://localhost:5000

### What to show in the UI

1. **Experiments sidebar** — click `loan_approval_experimentation` to see all test runs
2. **Runs table** — each row is a model. Columns show metrics (f1_score, accuracy, precision, recall, auc_roc) and parameters (hyperparameters)
3. **Compare runs** — select 2+ runs, click "Compare" to see metrics side by side with charts
4. **Single run** — click a run name to see:
   - Parameters tab: all hyperparameters used
   - Metrics tab: all logged metrics
   - Artifacts tab: confusion_matrix.json, the saved model
5. **Model Registry** — click "Models" in the top nav to see registered model versions

---

# Phase 2: Production

## 9. Promote Best Model

Takes the best model from experimentation, extracts its optimized hyperparameters from MLflow, and writes a production config file (`config/experiments_prod.yaml`).

Replace `NomeDoModelo` with the model name from the comparison table (e.g., `SklearnRF_Optimized`).

```bash
make promote MODEL=NomeDoModelo
```

## 10. Run Production Pipeline

Trains the promoted model in production mode and logs it to the `loan_approval_production` MLflow experiment.

```bash
make prod
```

## 11. Compare Production Results

Same as step 6, but for production runs only.

```bash
make compare ENV=prod
```

## 12. Analyze Production

Same analysis as step 7, but for the production model. Shows data, KPIs, feature importances, and per-sample decisions.

```bash
make analyze ENV=prod
```

## 13. Post-Deploy Check

Compares the current production run with the previous one. Checks 5 metrics (f1, accuracy, precision, recall, train_time). Flags as **DEGRADED** if f1 drops more than 0.02, otherwise reports **STABLE**.

```bash
make post-deploy
```

## 14. MLflow UI (Production)

Open MLflow again and switch to `loan_approval_production` in the sidebar to see the promoted model runs and compare with previous deployments.

```bash
make mlflow
```

---

# Phase 3: Inference

## 15. Serve

Starts an interactive CLI that loads the production model from MLflow. You paste JSON, it returns the prediction.

```bash
make serve
```

### Example Input

```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_emp_exp": 5,
  "loan_amnt": 10000,
  "loan_int_rate": 8.5,
  "loan_percent_income": 0.17,
  "cb_person_cred_hist_length": 5,
  "credit_score": 700,
  "person_gender": "female",
  "person_education": "Bachelor",
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "previous_loan_defaults_on_file": "No"
}
```

### Example Output

```
APPROVED (probability: 0.8234)
```

### More Examples to Try

**High risk profile (likely rejected):**

```json
{
  "person_age": 21,
  "person_income": 15000,
  "person_emp_exp": 0,
  "loan_amnt": 35000,
  "loan_int_rate": 18.5,
  "loan_percent_income": 2.33,
  "cb_person_cred_hist_length": 1,
  "credit_score": 450,
  "person_gender": "male",
  "person_education": "High School",
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "previous_loan_defaults_on_file": "Yes"
}
```

**Low risk profile (likely approved):**

```json
{
  "person_age": 45,
  "person_income": 120000,
  "person_emp_exp": 20,
  "loan_amnt": 5000,
  "loan_int_rate": 5.2,
  "loan_percent_income": 0.04,
  "cb_person_cred_hist_length": 18,
  "credit_score": 800,
  "person_gender": "female",
  "person_education": "Master",
  "person_home_ownership": "OWN",
  "loan_intent": "EDUCATION",
  "previous_loan_defaults_on_file": "No"
}
```

---

# Maintenance (Monthly Retraining)

```bash
make prod
make post-deploy
```

# Clean Everything

Removes all generated data, outputs, and MLflow history.

```bash
make clean
```
