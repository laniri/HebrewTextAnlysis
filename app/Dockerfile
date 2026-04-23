FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from app-specific requirements
COPY app/requirements.txt /tmp/app-requirements.txt
RUN pip install --no-cache-dir -r /tmp/app-requirements.txt && \
    rm /tmp/app-requirements.txt

# Copy the application code and ML module references
COPY app/ /app/app/
COPY ml/ /app/ml/
COPY hebrew_profiler/ /app/hebrew_profiler/

# Expose the API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
