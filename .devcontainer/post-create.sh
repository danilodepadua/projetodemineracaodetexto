#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$repo_root/.venv"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Creating project virtual environment at $venv_dir"
    python -m venv "$venv_dir"
else
    echo "Reusing project virtual environment at $venv_dir"
fi

python_version="$($venv_dir/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.12" ]]; then
    echo "Expected Python 3.12 in $venv_dir, found $python_version" >&2
    exit 1
fi

"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r "$repo_root/requirements.txt"
"$venv_dir/bin/python" -m pip check

echo "Project environment ready: $venv_dir/bin/python"
