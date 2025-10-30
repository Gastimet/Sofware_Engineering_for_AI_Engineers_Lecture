
FROM python:3.11-slim

# sistem bağımlılıkları (AutoGluon, LightGBM, CatBoost için yeterli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Build context is parent (..), so requirements.txt is at root
COPY requirements.txt /tmp/requirements.txt

# pip güncelle + requirements
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all source code
COPY . /workspace

EXPOSE 8000