# Makefile for the MammoScan AI Project

# --- Configuration ---
# Define our champion model and its parameters here.
# If we ever get a new champion, we only have to change these lines.
CHAMPION_MODEL_NAME = baseline
CHAMPION_MODEL_FILENAME = baseline_model_v2.keras
CHAMPION_MODEL_PATH = models/checkpoints/$(CHAMPION_MODEL_FILENAME)
CHAMPION_THRESHOLD = 0.110593
REPORTS_PATH = reports/champion_metrics.json

# --- Main Pipeline Commands ---
.PHONY: run-pipeline
run-pipeline: preprocess train evaluate
	@echo "✅ --- Full pipeline completed successfully! ---"

.PHONY: preprocess
preprocess:
	@echo "--- étape 1/3: Running data preprocessing ---"
	python -m ml.scripts.preprocess

.PHONY: train
train:
	@echo "--- étape 2/3: Training champion model ---"
	python -m ml.scripts.train \
		--model-name $(CHAMPION_MODEL_NAME) \
		--model-save-path $(CHAMPION_MODEL_PATH) \
		--epochs 20 # Using the epochs from the winning run

.PHONY: evaluate
evaluate:
	@echo "--- étape 3/3: Evaluating champion model ---"
	python -m ml.scripts.evaluate \
		--model-name $(CHAMPION_MODEL_NAME) \
		--model-path $(CHAMPION_MODEL_PATH) \
		--report-path $(REPORTS_PATH) \
		--threshold $(CHAMPION_THRESHOLD)

# --- Utility Commands ---
.PHONY: clean
clean:
	@echo "Cleaning up processed data and reports..."
	rm -rf data/processed
	rm -rf reports
	dvc checkout data/processed.dvc # Revert to the last versioned data

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  run-pipeline   Run the full preprocess -> train -> evaluate pipeline."
	@echo "  preprocess     Run only the data preprocessing script."
	@echo "  train          Train the current champion model."
	@echo "  evaluate       Evaluate the current champion model."
	@echo "  clean          Remove generated data and reports."