FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (MLSecOps)
RUN useradd -m -u 1000 appuser

WORKDIR /workspace

# Fix: Explicitly copy requirements from the project folder relative to the build context (..)
# The context is the parent folder, so we point to the subdirectory.
COPY Autogluon_multiple_regression_models/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all source code
COPY . /workspace

# Switch to non-root user
RUN chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8000