#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data/embeddings_db
mkdir -p /app/models/colqwen2/model
mkdir -p /app/models/colqwen2/processor

# Set proper permissions
chmod -R 777 /app/data
chmod -R 777 /app/models

# Execute the provided command (usually python app.py)
exec "$@"