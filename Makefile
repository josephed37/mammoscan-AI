# Makefile for the MammoScan AI Project

# --- Configuration ---
CHAMPION_MODEL_NAME = baseline
CHAMPION_MODEL_FILENAME = baseline_model_v2.keras
CHAMPION_MODEL_PATH = models/checkpoints/$(CHAMPION_MODEL_FILENAME)
CHAMPION_THRESHOLD = 0.110593
REPORTS_PATH = reports/champion_metrics.json

# --- Docker Configuration ---
COMPOSE_FILE = deployments/docker-compose.yml

# --- Main Pipeline Commands ---
.PHONY: run-pipeline
run-pipeline: preprocess train evaluate
	@echo "✅ --- Full pipeline completed successfully! ---"

.PHONY: preprocess
preprocess:
	@echo "--- 1/3: Running data preprocessing ---"
	python -m ml.scripts.preprocess

.PHONY: train
train:
	@echo "--- 2/3: Training champion model ---"
	python -m ml.scripts.train \
		--model-name $(CHAMPION_MODEL_NAME) \
		--model-save-path $(CHAMPION_MODEL_PATH) \
		--epochs 20

.PHONY: evaluate
evaluate:
	@echo "--- 3/3: Evaluating champion model ---"
	python -m ml.scripts.evaluate \
		--model-name $(CHAMPION_MODEL_NAME) \
		--model-path $(CHAMPION_MODEL_PATH) \
		--report-path $(REPORTS_PATH) \
		--threshold $(CHAMPION_THRESHOLD)

# --- Docker Commands ---
# Define the path to your docker-compose file
COMPOSE_FILE := deployments/docker-compose.yml

# --- Docker Commands ---
.PHONY: docker-build
docker-build:
	@echo "--- 🐳 Building Docker images ---"
	docker compose --project-directory . -f $(COMPOSE_FILE) build

.PHONY: docker-up
docker-up:
	@echo "--- 🚀 Starting all services with Docker Compose ---"
	docker compose --project-directory . -f $(COMPOSE_FILE) up -d

.PHONY: docker-down
docker-down:
	@echo "--- 🛑 Stopping all services ---"
	docker compose --project-directory . -f $(COMPOSE_FILE) down

.PHONY: docker-logs
docker-logs:
	@echo "--- 📜 Tailing logs for all services ---"
	docker compose --project-directory . -f $(COMPOSE_FILE) logs -f

# You can add a convenience target for rebuilding from scratch
.PHONY: docker-rebuild
docker-rebuild: docker-down
	@echo "--- 🔄 Rebuilding Docker images from scratch ---"
	docker compose --project-directory . -f $(COMPOSE_FILE) build --no-cache
	$(MAKE) docker-up

# --- Utility Commands ---
.PHONY: clean
clean:
	@echo "Cleaning up processed data and reports..."
	rm -rf data/processed
	rm -rf reports
	dvc checkout data/processed.dvc

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  run-pipeline   Run the full preprocess -> train -> evaluate pipeline."
	@echo "  docker-build   Build all Docker images for the application."
	@echo "  docker-up      Start the application stack."
	@echo "  docker-down    Stop the application stack."
	@echo "  docker-logs    View logs from running services."
	@echo "  clean          Remove generated data and reports."