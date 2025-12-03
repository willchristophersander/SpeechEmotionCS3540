# Dockerfile for Fly.io deployment
# Uses Python 3.10 for compatibility

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Quantize model during build (if not already quantized)
RUN python quantize_model.py || echo "Quantization skipped (model may already be quantized)"

# Expose port
EXPOSE 8080

# Health check (using curl which is available in slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run gunicorn
# Use PORT environment variable (Fly.io sets this)
# Using shell form to allow environment variable expansion
CMD gunicorn app:app --timeout 120 --workers 1 --threads 2 --bind 0.0.0.0:${PORT:-8080}

