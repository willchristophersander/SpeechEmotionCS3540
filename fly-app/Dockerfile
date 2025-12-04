# Dockerfile for Fly.io deployment
# Uses Python 3.10 for compatibility
# Optimized for minimal disk space usage

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
# Build dependencies (gcc, g++) are removed after pip install to save space
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install CPU-only PyTorch first (much smaller, ~200MB vs ~2GB with CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove build dependencies to save space (gcc, g++ no longer needed)
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Quantize model during build (if not already quantized)
# This can take 5-10 minutes - consider pre-quantizing locally to speed up builds
# To pre-quantize: run 'python quantize_model.py' locally and commit the quantized model
RUN if [ ! -f checkpoints/6class/crnn_multi_dataset_6class_quantized.pth ]; then \
        echo "Quantizing 6-class model (this may take 5-10 minutes)..." && \
        python quantize_model.py; \
    else \
        echo "Quantized model already exists, skipping quantization"; \
    fi

# Clean up Python cache
RUN find /usr/local/lib/python3.10 -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -name "*.pyc" -delete 2>/dev/null || true

# Expose port
EXPOSE 8080

# Health check (using curl which is available in slim image)
# Use fixed port 8080 since PORT env var may not be available in healthcheck context
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run gunicorn
# Fly.io sets PORT=8080 in fly.toml, but we'll use exec form with explicit port for reliability
# Using exec form is more efficient than shell form
CMD ["gunicorn", "app:app", "--timeout", "120", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:8080"]

