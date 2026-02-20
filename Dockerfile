# Dockerfile for Streamlit app and/or FastAPI inference API
# Deploy to Azure Container Instances or App Service
#
# Build for Streamlit only (default):
#   docker build -t seg-app .
# Build for FastAPI API only:
#   docker build --build-arg APP_MODE=api -t seg-api .
# Build for both (Streamlit + API in same container):
#   docker build --build-arg APP_MODE=both -t seg-both .
# Run both: docker run -p 8501:8501 -p 8000:8000 seg-both

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (OpenCV + build tools for C extensions e.g. psutil)
# libgl1 replaces deprecated libgl1-mesa-glx on Debian Bookworm/Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip for better wheel support, then install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY app/ app/

# Entrypoint to run Streamlit, API, or both
RUN chmod +x app/docker-entrypoint.sh

# Choose app mode: "streamlit" (default), "api", or "both"
ARG APP_MODE=streamlit
ENV APP_MODE=${APP_MODE}

EXPOSE 8501 8000

ENTRYPOINT ["/app/app/docker-entrypoint.sh"]
