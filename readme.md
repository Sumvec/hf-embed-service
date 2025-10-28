# hf-embed-service — Local run instructions

This repository contains a FastAPI service that is normally containerized. These instructions show how to run it locally (no Docker) using Python 3.11 on macOS.

Prerequisites
- Python 3.11 installed (Homebrew or pyenv recommended).
  - Homebrew: `brew install python@3.11`
  - pyenv: `pyenv install 3.11.x && pyenv local 3.11.x`
- git and a terminal.

Quick start (recommended)
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Upgrade packaging tools and install dependencies:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Download NLTK tokenizer data (punkt):
```bash
python -m nltk.downloader punkt
```

4. Configure environment variables:
- Copy `.env.example` -> `.env` (if present) and edit values, or export env vars directly:
```bash
cp .env.example .env
# edit .env with your editor, OR
export SOME_CONFIG=value
```
The app uses standard env vars (or library such as python-dotenv / pydantic for config).

5. Run the FastAPI app with uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/docs for the interactive OpenAPI UI.

Notes & tips
- If Python 3.11 is not available, use pyenv or Homebrew to install it.
- If you need GPU support (PyTorch), install a matching torch wheel manually — see https://pytorch.org for the correct command for your CUDA version. Example CPU-only install:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- For reproducible installs, pin versions in `requirements.txt`.
- If you run into build issues (compilation of native packages), install macOS build tools:
  ```bash
  xcode-select --install
  brew install pkg-config libsndfile
  ```
- If you prefer to run inside Docker, see the provided `Dockerfile` (already uses python:3.11-slim).

Common troubleshooting
- "python3.11: command not found": install via Homebrew or use pyenv.
- Permission errors on pip: ensure virtualenv is activated or add `--user` (not recommended for project installs).
- Slow installs for heavy packages (torch, transformers): prefer prebuilt wheels or use `pip download` on a machine with good bandwidth.

That's all — the app should now be running locally on port 8000.