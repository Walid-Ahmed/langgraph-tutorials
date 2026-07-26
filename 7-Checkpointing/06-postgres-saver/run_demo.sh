#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root from this script's location so the demo works
# regardless of the terminal's current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Missing Python virtual environment: $REPO_ROOT/.venv"
    echo "Create it with:"
    echo "  python3 -m venv \"$REPO_ROOT/.venv\""
    echo "  source \"$REPO_ROOT/.venv/bin/activate\""
    echo "  pip install -r \"$REPO_ROOT/requirements.txt\""
    exit 1
fi

# .venv contains this repository's Python interpreter and installed packages.
# It does not contain the PostgreSQL server or database.
source "$VENV_ACTIVATE"
cd "$REPO_ROOT"

if command -v pg_isready >/dev/null 2>&1 && ! pg_isready >/dev/null 2>&1; then
    echo "PostgreSQL is not accepting connections."
    echo "For the Homebrew PostgreSQL 16 installation, start it with:"
    echo "  brew services start postgresql@16"
    exit 1
fi

echo "1/3 Creating or validating LangGraph checkpoint tables..."
python "7-Checkpointing/06-postgres-saver/00_setup_tables.py"

echo "2/3 Saving the first conversation turn..."
python "7-Checkpointing/06-postgres-saver/01_save_name.py"

echo "3/3 Starting a new process and recalling the saved conversation..."
python "7-Checkpointing/06-postgres-saver/02_recall_name.py"

echo "PostgreSQL checkpoint demo completed."
