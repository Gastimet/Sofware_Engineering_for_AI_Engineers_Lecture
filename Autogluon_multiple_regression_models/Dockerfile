FROM python:3.11-slim

# sistem bağımlılıkları (AutoGluon, LightGBM, CatBoost için yeterli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Since build context is parent (..), copy requirements from there
COPY requirements.txt /workspace/requirements.txt

# pip güncelle + requirements
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# proje dosyaları (tüm parent directory)
COPY . /workspace

EXPOSE 8000