@echo off
:: lite-assistant launch script for Windows.
:: Creates a venv, installs the package, downloads any missing models, and
:: starts the server. Arguments go to 'lite-assistant serve', e.g. --port 8000.

cd /d "%~dp0"

IF NOT EXIST ".venv" (
    echo Creating python virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

:: Always sync dependencies so new engines are installed automatically
pip install -e .

:: Engines skip models that are already downloaded
lite-assistant setup

lite-assistant serve %*
