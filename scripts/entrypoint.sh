#!/bin/bash

echo "=== Hebrew Writing Coach Entrypoint ==="
echo "MODEL_PATH: ${MODEL_PATH:-/app/model}"
echo "MODEL_S3_BUCKET: ${MODEL_S3_BUCKET:-NOT SET}"
echo "MODEL_S3_KEY: ${MODEL_S3_KEY:-NOT SET}"

MODEL_DIR="${MODEL_PATH:-/app/model}"
mkdir -p "$MODEL_DIR"

# Download model from S3 if not already present
if [ ! -f "$MODEL_DIR/model.pt" ]; then
    if [ -z "$MODEL_S3_BUCKET" ] || [ -z "$MODEL_S3_KEY" ]; then
        echo "ERROR: MODEL_S3_BUCKET or MODEL_S3_KEY not set. Cannot download model."
    else
        # Strip trailing slash from key to avoid double-slash
        S3_KEY="${MODEL_S3_KEY%/}"
        echo "Downloading model from s3://${MODEL_S3_BUCKET}/${S3_KEY}/ ..."
        aws s3 sync "s3://${MODEL_S3_BUCKET}/${S3_KEY}/" "$MODEL_DIR/" 2>&1
        SYNC_EXIT=$?
        echo "aws s3 sync exit code: $SYNC_EXIT"
        if [ $SYNC_EXIT -eq 0 ] && [ -f "$MODEL_DIR/model.pt" ]; then
            echo "Model downloaded successfully ($(ls -lh $MODEL_DIR/model.pt | awk '{print $5}'))"
        else
            echo "ERROR: Model download failed or model.pt not found after download"
            ls -la "$MODEL_DIR/" 2>&1 || true
        fi
    fi
else
    echo "Model already present at $MODEL_DIR ($(ls -lh $MODEL_DIR/model.pt | awk '{print $5}'))"
fi

echo "=== Starting supervisor ==="
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
