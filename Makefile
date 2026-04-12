# ─── CI/CD Pipeline (simulated) ──────────────────────────
#
# Full deployment flow:
#   make setup               install dependencies (once)
#   make configs             validate all configs (gate)
#   make test                train all models in experimentation
#   make compare ENV=test    compare results, pick best
#   make compare ENV=prod    compare production results
#   make promote MODEL=Name  move best to production
#   make prod                train production model
#   make analyze ENV=test    full analysis with KPIs and feature importances
#   make analyze ENV=prod    same for production
#   make post-deploy         compare current vs previous production run
#   make mlflow              open MLflow UI
#   make serve               start inference service
#
# Maintenance flow (monthly):
#   make prod                retrain with new data
#   make post-deploy         check if model degraded
#
# ──────────────────────────────────────────────────────────

.PHONY: setup configs test prod compare promote post-deploy analyze mlflow serve clean

# colors
CYAN    := \033[1;36m
GREEN   := \033[1;32m
YELLOW  := \033[1;33m
MAGENTA := \033[1;35m
DIM     := \033[0;37m
RESET   := \033[0m

setup:
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(CYAN)  SETUP                                                    $(RESET)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Creating a Python virtual environment (.venv) and        $(RESET)"
	@echo "$(DIM)  installing all project dependencies from requirements.txt$(RESET)"
	@echo ""
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "$(GREEN)  DONE. Run: source .venv/bin/activate$(RESET)"
	@echo ""

configs:
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(CYAN)  CONFIGS                                                  $(RESET)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Validating all YAML configuration files: data contract,  $(RESET)"
	@echo "$(DIM)  pipeline settings, quality rules, and experiment configs  $(RESET)"
	@echo "$(DIM)  for both experimentation and production modes.            $(RESET)"
	@echo "$(DIM)  This is a gate — nothing runs if configs are broken.     $(RESET)"
	@echo ""
	.venv/bin/python -c "from src.config import DataConfig, PipelineConfig, QualityConfig, ExperimentsConfig; DataConfig.load(); PipelineConfig.load(); QualityConfig.load(); ExperimentsConfig.load(mode='production'); ExperimentsConfig.load(mode='experimentation'); print('all configs ok')"

test:
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  EXPERIMENTATION                                         $(RESET)"
	@echo "$(GREEN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Running the full ML pipeline in experimentation mode.    $(RESET)"
	@echo "$(DIM)  Trains 10 models and logs everything to MLflow.          $(RESET)"
	@echo "$(DIM)                                                            $(RESET)"
	@echo "$(DIM)  6 jobs in sequence:                                      $(RESET)"
	@echo "$(DIM)    1. Ingestion  — download dataset, convert to Parquet   $(RESET)"
	@echo "$(DIM)    2. Quality    — null checks, duplicates, value ranges  $(RESET)"
	@echo "$(DIM)    3. Train      — preprocess + train all models          $(RESET)"
	@echo "$(DIM)    4. Selection  — rank by f1, financial impact analysis  $(RESET)"
	@echo "$(DIM)    5. Inference  — test prediction with the best model    $(RESET)"
	@echo "$(DIM)    6. Drift      — check for statistical drift           $(RESET)"
	@echo ""
	.venv/bin/python run_pipeline.py experimentation

prod:
	@echo ""
	@echo "$(MAGENTA)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(MAGENTA)  PRODUCTION                                               $(RESET)"
	@echo "$(MAGENTA)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Running the full pipeline in production mode.             $(RESET)"
	@echo "$(DIM)  Trains only the promoted model and registers it in MLflow.$(RESET)"
	@echo "$(DIM)  Same 6 jobs as experimentation, but with 1 model only.   $(RESET)"
	@echo ""
	.venv/bin/python run_pipeline.py production $(ID)

compare:
	@test -n "$(ENV)" || (echo "usage: make compare ENV=test|prod" && exit 1)
	@echo ""
	@echo "$(YELLOW)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(YELLOW)  COMPARE ($(ENV))                                          $(RESET)"
	@echo "$(YELLOW)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Queries MLflow and displays all models ranked by f1_score.$(RESET)"
	@echo "$(DIM)  Shows accuracy, precision, recall, overfitting gap,      $(RESET)"
	@echo "$(DIM)  training time, and AUC-ROC. Exports results to CSV.      $(RESET)"
	@echo ""
	.venv/bin/python scripts/compare.py $(ENV)

promote:
	@echo ""
	@echo "$(MAGENTA)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(MAGENTA)  PROMOTE: $(MODEL)                                        $(RESET)"
	@echo "$(MAGENTA)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Takes the selected model from experimentation, extracts  $(RESET)"
	@echo "$(DIM)  its optimized hyperparameters from MLflow, and writes    $(RESET)"
	@echo "$(DIM)  the production config (config/experiments_prod.yaml).    $(RESET)"
	@echo ""
	.venv/bin/python scripts/promote.py $(MODEL)

post-deploy:
	@echo ""
	@echo "$(YELLOW)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(YELLOW)  POST-DEPLOY                                              $(RESET)"
	@echo "$(YELLOW)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Compares the current production run with the previous    $(RESET)"
	@echo "$(DIM)  one. Checks f1, accuracy, precision, recall, and train   $(RESET)"
	@echo "$(DIM)  time. Flags DEGRADED if f1 drops more than 0.02.        $(RESET)"
	@echo ""
	.venv/bin/python scripts/post_deploy.py

analyze:
	@test -n "$(ENV)" || (echo "usage: make analyze ENV=test|prod" && exit 1)
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(CYAN)  ANALYZE ($(ENV))                                          $(RESET)"
	@echo "$(CYAN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Full analysis report: data samples, KPIs in percentages, $(RESET)"
	@echo "$(DIM)  approved vs rejected profiles, feature importances from  $(RESET)"
	@echo "$(DIM)  the best model, and per-sample reasoning showing which   $(RESET)"
	@echo "$(DIM)  factors drove each approval or rejection decision.       $(RESET)"
	@echo ""
	.venv/bin/python scripts/analyze.py $(ENV)

mlflow:
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  MLFLOW UI                                                $(RESET)"
	@echo "$(GREEN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Starting MLflow dashboard. Open http://localhost:5000    $(RESET)"
	@echo "$(DIM)  in your browser to explore experiments, compare runs,    $(RESET)"
	@echo "$(DIM)  view metrics/parameters, and check model registry.      $(RESET)"
	@echo ""
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

serve:
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  SERVE                                                    $(RESET)"
	@echo "$(GREEN)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Loading the production model from MLflow for inference.  $(RESET)"
	@echo "$(DIM)  Paste a JSON payload to get a prediction (APPROVED or    $(RESET)"
	@echo "$(DIM)  REJECTED with probability). Press Ctrl+C to exit.       $(RESET)"
	@echo ""
	.venv/bin/python -m src.serve

clean:
	@echo ""
	@echo "$(YELLOW)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(YELLOW)  CLEAN                                                    $(RESET)"
	@echo "$(YELLOW)════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(DIM)  Removing all generated data, outputs, MLflow database,  $(RESET)"
	@echo "$(DIM)  and run history. Start fresh with make setup.           $(RESET)"
	@echo ""
	rm -rf data/raw/* data/processed/* outputs/ mlflow.db mlruns/
	@echo "$(GREEN)  DONE.$(RESET)"
	@echo ""
