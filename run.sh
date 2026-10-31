#!/bin/bash
# lite-assistant launch script for Linux/macOS.
# Creates a venv, installs the package, downloads any missing models, and
# starts the server. Arguments go to 'lite-assistant serve', e.g. --port 8000.

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Always sync dependencies so new engines are installed automatically
pip install -e .
# Engines skip models that are already downloaded
lite-assistant setup

lite-assistant serve "$@"
