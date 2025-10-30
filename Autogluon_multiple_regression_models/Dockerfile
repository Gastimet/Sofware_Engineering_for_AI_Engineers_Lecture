FROM python:3.11-slim

# sistem bağımlılıkları (AutoGluon, LightGBM, CatBoost için yeterli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements from current context (Autogluon_multiple_regression_models/)
COPY requirements.txt /workspace/requirements.txt

# pip güncelle + requirements
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy only necessary project files (we'll mount the full source via volumes)
# This layer won't change often, so it's cached
COPY . /workspace/project_files

EXPOSE 8000