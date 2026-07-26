#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root so this launcher works from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Missing virtual environment: $REPO_ROOT/.venv"
    echo "Create it and install requirements first:"
    echo "  python3 -m venv \"$REPO_ROOT/.venv\""
    echo "  source \"$REPO_ROOT/.venv/bin/activate\""
    echo "  pip install -r \"$REPO_ROOT/requirements.txt\""
    exit 1
fi

source "$VENV_ACTIVATE"
cd "$REPO_ROOT"

if command -v pg_isready >/dev/null 2>&1 && ! pg_isready >/dev/null 2>&1; then
    echo "PostgreSQL is not accepting connections."
    echo "For Homebrew PostgreSQL 16, try:"
    echo "  brew services start postgresql@16"
    exit 1
fi

echo "1/3 Creating or validating PostgresStore tables..."
python "8-Long-Term-Memory/03-postgres-store/00_setup_tables.py"

echo "2/3 Saving a user profile in process 1..."
python "8-Long-Term-Memory/03-postgres-store/01_save_profile.py"

echo "3/3 Reading the profile in process 2..."
python "8-Long-Term-Memory/03-postgres-store/02_read_profile.py"

echo "PostgresStore durability demo completed."
