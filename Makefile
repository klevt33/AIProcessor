# ───────────────────────────────────────────────────────
# Makefile for Docker build/deploy with environments and parameters
# ───────────────────────────────────────────────────────
# Example usage
# =============
# make build ENV=dev
# make deploy ENV=dev
# make all ENV=prod
# ─────────────────────────────────────────────────

# Configurable variables (override as needed)
ENV            ?= dev                 # Environment: dev or prod
CONFIG_FILE    := config.yaml         # YAML file storing APP_VERSION and ENVIRONMENT

# Remove accidental whitespace from ENV value
ENV := $(strip $(ENV))

# Extract version from config.yaml
CONTAINER_VERSION := 1

# Map environment to ACR name, registry, image name
ifeq ($(ENV),dev)
	ACR_NAME   := aksrpaaicrdev
	REGISTRY   := $(ACR_NAME).azurecr.io
	IMAGE_NAME := aks-wa-sr-dev
else ifeq ($(ENV),prod)
	ACR_NAME   := aksrpaaicrprod
	REGISTRY   := $(ACR_NAME).azurecr.io
	IMAGE_NAME := aks-wa-sr-prod
else
	$(error Invalid ENV "$(ENV)". Valid options: dev, prod)
endif

# Full image reference, e.g. aksrpaaicrdev.azurecr.io/aks-wa-sr-dev:v0.0.3
IMAGE := $(REGISTRY)/$(IMAGE_NAME):v$(CONTAINER_VERSION)

# Default goal
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help:
	@echo ""
	@echo "Usage:"
	@echo "  make build [ENV=dev|prod]        # Update config.yaml & build image"
	@echo "  make deploy [ENV=dev|prod]       # Login, push & deploy image"
	@echo "  make all [ENV=dev|prod]          # build + deploy"
	@echo ""
	@echo "Override ENV, e.g.: make build ENV=prod"
	@echo ""

# Bump APP_VERSION and set ENVIRONMENT to 'web' in config.yaml using PowerShell
.PHONY: bump
bump:
	@powershell -NoProfile -Command "$$lines = Get-Content '$(CONFIG_FILE)'; \
	$$out = foreach ($$line in $$lines) { \
		if ($$line -match '^APP_VERSION:\s*([0-9]+)\.([0-9]+)\.([0-9]+)$$') { \
			$$maj, $$min, $$patch = $$Matches[1], $$Matches[2], $$Matches[3]; \
			$$newVer = \"$$maj.$$min.$$([int]$$patch + 1)\"; \
			\"APP_VERSION: $$newVer\"; \
		} elseif ($$line -match '^ENVIRONMENT:') { \
			\"ENVIRONMENT: web\"; \
		} else { \
			$$line; \
		} \
	}; \
	$$out | Set-Content '$(CONFIG_FILE)';"
	@powershell -NoProfile -Command "Write-Host 'Updated APP_VERSION to' ((Select-String -Pattern '^APP_VERSION:' -Path '$(CONFIG_FILE)').Matches[0].Groups[1].Value) 'and ENVIRONMENT set to web'"

# Build target
.PHONY: build
build: bump
	@echo "Building Docker image $(IMAGE) on environment $(ENV)..."
	@docker build --no-cache \
		--build-arg APP_ENV=$(ENV) \
		--build-arg APP_VERSION=$(shell powershell -NoProfile -Command "(Select-String -Pattern '^APP_VERSION:' -Path $(CONFIG_FILE)).Matches[0].Groups[1].Value") \
		-t $(IMAGE) \
		.

# Deploy target
.PHONY: deploy
deploy:
	@echo "Logging in to ACR $(ACR_NAME)..."
	@az acr login --name $(ACR_NAME)
	@echo "Pushing image $(IMAGE)..."
	@docker push $(IMAGE)
	@echo "Deploying image $(IMAGE) to $(ENV)..."
	@echo "Deployed $(IMAGE)"

# Shortcut for build + deploy
.PHONY: all build-deploy
all: build deploy

build-deploy: all

# Possible warning...
# make: warning:  Clock skew detected.  Your build may be incomplete.
# Run below command before running make command to avoid that warning.
# touch Makefile
