# ===== Stage 1: Builder =====
FROM python:3.11-slim as builder

# Set work directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch first (smaller size)
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install to a specific directory
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== Stage 2: Runtime =====
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Download NLTK tokenizer data (punkt)
RUN python -m nltk.downloader punkt

# Copy app code
COPY app ./app

# Copy .env if you want (optional, can mount at runtime)
COPY .env.example .env

# Expose port
EXPOSE 8002

# Start FastAPI using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]