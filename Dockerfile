FROM tensorflow/tensorflow:2.17.0

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install remaining deps with higher timeout (split to avoid cache bust on large ones)
# Small/fast ones first
RUN pip install --no-cache-dir --timeout 120 \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.9.2 \
    numpy==1.26.4 \
    joblib==1.4.2 \
    scikit-learn==1.6.1 \
    pandas==2.2.3 \
    python-dateutil==2.9.0.post0 \
    nltk==3.8.1

# Larger Google Cloud packages separately
RUN pip install --no-cache-dir --timeout 120 \
    google-cloud-aiplatform==1.67.1 \
    google-cloud-secret-manager==2.20.0

COPY . .
RUN chmod -R 755 /app/models

EXPOSE 8080

# Use Gunicorn + Uvicorn for production stability
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:${PORT:-8080}", "--workers", "1"]
