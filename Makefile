# ─── CI/CD Pipeline (simulated) ──────────────────────────
#
# Full deployment flow:
#   make setup          install dependencies (once)
#   make configs        validate all configs (gate)
#   make test           train all models in experimentation
#   make compare        compare results, pick best
#   make promote MODEL=Name  move best to production
#   make prod           train production model
#   make compare-prod   verify production results
#   make post-deploy    compare current vs previous production run
#   make serve          start inference service
#
# Maintenance flow (monthly):
#   make prod           retrain with new data
#   make post-deploy    check if model degraded
#
# ──────────────────────────────────────────────────────────

.PHONY: setup configs test prod compare compare-prod promote post-deploy mlflow serve clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

configs:
	.venv/bin/python -c "from src.config import DataConfig, PipelineConfig, QualityConfig, ExperimentsConfig; DataConfig.load(); PipelineConfig.load(); QualityConfig.load(); ExperimentsConfig.load(mode='production'); ExperimentsConfig.load(mode='experimentation'); print('all configs ok')"

test:
	.venv/bin/python run_pipeline.py experimentation

prod:
	.venv/bin/python run_pipeline.py production $(ID)

compare:
	.venv/bin/python scripts/compare.py loan_approval_experimentation

compare-prod:
	.venv/bin/python scripts/compare.py loan_approval_production

promote:
	.venv/bin/python scripts/promote.py $(MODEL)

post-deploy:
	.venv/bin/python scripts/post_deploy.py

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

serve:
	.venv/bin/python -m src.serve

clean:
	rm -rf data/raw/* data/processed/* outputs/ mlflow.db mlruns/
