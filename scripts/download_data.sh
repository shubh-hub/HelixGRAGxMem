#!/bin/bash

# This script downloads the necessary raw data for Phase 1 of the HelixRAGxMem project.

echo "--- Starting Data Acquisition for HelixRAGxMem ---"

# --- Create Directories (redundant if already run, but safe) ---
echo "[1/3] Ensuring data directories exist..."
mkdir -p data/raw data/processed vendor
echo "Done."

# --- Download Hetionet ---
echo "[2/3] Downloading Hetionet v1.0..."
HETIONET_URL="https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2"
HETIONET_BZ2="data/raw/hetionet-v1.0.json.bz2"
HETIONET_JSON="data/raw/hetionet-v1.0.json"

if [ -f "$HETIONET_JSON" ]; then
    echo "Hetionet JSON already exists. Skipping download."
else
    curl -L -o "$HETIONET_BZ2" "$HETIONET_URL"
    echo "Decompressing Hetionet data..."
    bunzip2 "$HETIONET_BZ2"
fi
echo "Done."

# --- Clone KG-LLM-Bench ---
echo "[3/3] Cloning KG-LLM-Bench repository..."
KG_BENCH_DIR="vendor/LLM-KG-Bench"

if [ -d "$KG_BENCH_DIR" ]; then
    echo "KG-LLM-Bench repository already exists. Skipping clone."
else
    git clone https://github.com/AKSW/LLM-KG-Bench.git "$KG_BENCH_DIR"
fi
echo "Done."

echo "--- Data Acquisition Complete ---"
