# =============================================================================
# Combined Dockerfile for Hebrew Writing Coach
# Builds React frontend + Python backend in a single container
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build the React frontend
# ---------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python backend + nginx + supervisor
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies: nginx, supervisor, curl (health check), awscli
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx \
        supervisor \
        curl \
        gcc \
        g++ \
        awscli \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (keeps image smaller — no CUDA needed for inference)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY app/requirements.txt /tmp/app-requirements.txt
COPY ml/requirements.txt /tmp/ml-requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/app-requirements.txt \
    -r /tmp/ml-requirements.txt \
    && rm /tmp/app-requirements.txt /tmp/ml-requirements.txt

# Copy application code
COPY app/ /app/app/
COPY ml/ /app/ml/
COPY analysis/ /app/analysis/
COPY hebrew_profiler/ /app/hebrew_profiler/

# Copy frontend build output
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html

# Copy nginx and supervisor configs
COPY deploy/nginx-prod.conf /etc/nginx/conf.d/default.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Remove default nginx site
RUN rm -f /etc/nginx/sites-enabled/default

# Copy entrypoint script
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create model directory
RUN mkdir -p /app/model

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost/api/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
