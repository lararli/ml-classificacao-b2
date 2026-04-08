"""
Promote a model from experimentation to production.
Copies the model entry from experiments_test.yaml to experiments_prod.yaml.

Usage: python promote.py MODEL_NAME
Example: python promote.py SklearnRFOptimized
"""

import sys
import yaml

if len(sys.argv) < 2:
    print("usage: python promote.py MODEL_NAME")
    print("example: python promote.py SklearnRFOptimized")
    sys.exit(1)

model_name = sys.argv[1]

# Load test experiments
with open("config/experiments_test.yaml") as f:
    test_config = yaml.safe_load(f)

# Find the model
model_entry = None
for m in test_config["models"]:
    if m["name"] == model_name:
        model_entry = m.copy()
        break

if not model_entry:
    print(f"model '{model_name}' not found in experiments_test.yaml")
    print(f"available: {[m['name'] for m in test_config['models']]}")
    sys.exit(1)

# Remove search_params (production uses fixed params)
if "search_params" in model_entry:
    del model_entry["search_params"]
    print(f"note: search_params removed (production uses fixed params)")

# Write to prod
prod_config = {"models": [model_entry]}

with open("config/experiments_prod.yaml", "w") as f:
    f.write("# experiments_prod.yaml\n")
    f.write("# Production models. Executed in the monthly pipeline.\n\n")
    yaml.dump(prod_config, f, default_flow_style=False, sort_keys=False)

print(f"\n{'='*60}")
print(f"PROMOTED: {model_name} -> production")
print(f"{'='*60}")
print(f"\nconfig/experiments_prod.yaml updated:")

with open("config/experiments_prod.yaml") as f:
    print(f.read())

print(f"next step: make prod")
