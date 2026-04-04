#!/usr/bin/env bash
# Start an ollama model server on a Slurm GPU node.
# Usage: slurm-serve-ollama.sh <model> [port] [num_gpus]
#
# Required env:
#   OLLAMA_MODELS   path where ollama stores model weights on the GPU node
#                   (set in your ~/.bashrc or a project .env file)
#
# Prints the base URL once the server is ready:
#   http://<node>:<port>/v1
#
# The job runs until killed. To stop: scancel <jobid>
# The URL is also written to /tmp/ollama-server-<model>.url

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [port] [num_gpus]}"
PORT="${2:-11434}"
NUM_GPUS="${3:-2}"
OLLAMA="$HOME/.local/bin/ollama"
LOGFILE="/tmp/ollama-slurm-${MODEL//\//-}.log"
URLFILE="/tmp/ollama-server-${MODEL//\//-}.url"

MODELS_DIR="${OLLAMA_MODELS:?OLLAMA_MODELS must be set (e.g. export OLLAMA_MODELS=/scratch/you/ollama-models)}"

if [[ ! -x "$OLLAMA" ]]; then
    echo "ERROR: ollama not found at $OLLAMA" >&2
    exit 1
fi

# Check if a server is already up for this model
if [[ -f "$URLFILE" ]]; then
    URL=$(cat "$URLFILE")
    NODE=$(echo "$URL" | sed 's|http://||;s|:.*||')
    if curl -sf "http://${NODE}:${PORT}/api/tags" >/dev/null 2>&1; then
        echo "already running: $URL"
        exit 0
    fi
    rm -f "$URLFILE"
fi

# Submit Slurm job
JOBID=$(sbatch --parsable \
    --partition=compregular \
    --gres=gpu:a100:${NUM_GPUS} \
    --cpus-per-task=8 \
    --mem=64G \
    --time=8:00:00 \
    --job-name="ollama-${MODEL//\//-}" \
    --output="$LOGFILE" \
    --error="$LOGFILE" \
    --wrap="
        export OLLAMA_MODELS=${MODELS_DIR}
        export OLLAMA_HOST=0.0.0.0:${PORT}
        export OLLAMA_KEEP_ALIVE=24h
        export OLLAMA_MAX_LOADED_MODELS=3
        mkdir -p \$OLLAMA_MODELS
        $OLLAMA serve &
        SERVE_PID=\$!
        sleep 5
        $OLLAMA pull ${MODEL} 2>&1
        echo 'MODEL_READY'
        wait \$SERVE_PID
    ")

echo "Submitted job $JOBID, waiting for node allocation and model pull..." >&2

# Wait for job to start and get the node name
for i in $(seq 1 120); do
    STATE=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
    if [[ "$STATE" == "RUNNING" ]]; then
        NODE=$(squeue -j "$JOBID" -h -o "%N" 2>/dev/null)
        break
    elif [[ -z "$STATE" ]]; then
        echo "ERROR: job $JOBID disappeared" >&2
        exit 1
    fi
    sleep 5
done

if [[ -z "${NODE:-}" ]]; then
    echo "ERROR: timed out waiting for job to start" >&2
    exit 1
fi

echo "Running on $NODE, waiting for server + model pull..." >&2

# Wait for ollama to be ready (model pull can take a while first time)
for i in $(seq 1 180); do
    if curl -sf "http://${NODE}:${PORT}/api/tags" >/dev/null 2>&1; then
        URL="http://${NODE}:${PORT}/v1"
        echo "$URL" | tee "$URLFILE"
        echo "Job ID: $JOBID (stop with: scancel $JOBID)" >&2
        exit 0
    fi
    sleep 10
done

echo "ERROR: server on $NODE:$PORT did not become ready in time" >&2
echo "Check log: $LOGFILE" >&2
exit 1
