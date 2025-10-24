# ===== Stage 1: Base =====
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK tokenizer data (punkt)
RUN python -m nltk.downloader punkt

# Copy app code
COPY app ./app

# Copy .env if you want (optional, can mount at runtime)
COPY .env .env

# Expose port
EXPOSE 8000

# Start FastAPI using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]