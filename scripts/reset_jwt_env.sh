#!/usr/bin/env bash

# Usage:
#   source scripts/reset_jwt_env.sh
#
# Notes:
# - Must be sourced to affect the current shell environment.
# - Unsets DIGITALDOSSIER_API_TOKEN in this shell and clears macOS launchctl env.
# - Rebuilds & restarts backend + langgraph-worker to pick up .env with override.
# - Prints masked OPENAI_API_KEY inside the worker and confirms JWT auth mode.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced to modify your current shell."
  echo "Run: source scripts/reset_jwt_env.sh"
  exit 1
fi

echo "Unsetting DIGITALDOSSIER_API_TOKEN in current shell..."
unset DIGITALDOSSIER_API_TOKEN
echo "DIGITALDOSSIER_API_TOKEN='${DIGITALDOSSIER_API_TOKEN}' (expected empty)"

echo "Attempting to unset macOS persistent env (launchctl)..."
launchctl unsetenv DIGITALDOSSIER_API_TOKEN 2>/dev/null || true

echo "Restarting services with docker compose (backend, langgraph-worker)..."
docker compose down
docker compose up -d --build backend langgraph-worker

echo "Verifying OPENAI_API_KEY inside langgraph-worker (masked)..."
docker compose exec langgraph-worker sh -lc 'python - <<"PY"\nimport os\nk=os.getenv("OPENAI_API_KEY","" )\nprint("OPENAI_API_KEY:", (k[:8]+"…"+k[-4:]) if k else "NOT SET")\nPY'

echo "Checking publisher logs for JWT auth mode..."
docker compose logs --no-color langgraph-worker | grep -E "Using JWT authentication" | tail -n 5 || true

echo "Done. Start a new pipeline run from the UI to validate end-to-end."

