FROM python:3.11-slim

WORKDIR /app

# System deps (OpenCV + TensorFlow requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  gcc \
  libpython3-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better cache)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/backend/api.py api.py

# Ensure api and src are importable
ENV PYTHONPATH=/app

# 🔥 TensorFlow optimizations
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV OMP_NUM_THREADS=2
ENV TF_NUM_INTRAOP_THREADS=2
ENV TF_NUM_INTEROP_THREADS=2

EXPOSE 8000

# IMPORTANT: 1 worker only
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
