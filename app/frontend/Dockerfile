FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  gcc \
  libpython3-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/frontend/streamlit.py streamlit.py

EXPOSE 8501

CMD ["streamlit", "run", "/app/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
