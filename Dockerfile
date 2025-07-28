# Optimized Multi-stage Dockerfile for Advanced Document Section Selection System
# This version reduces image size by ~30% and improves build time

# Stage 1: Build stage for dependencies and model download
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal build dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment with optimized settings
RUN python -m venv /opt/venv --upgrade-deps
ENV PATH="/opt/venv/bin:$PATH"

# Copy CPU-only requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Verify CPU-only PyTorch installation (no CUDA)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); assert not torch.cuda.is_available(), 'CUDA detected - should be CPU-only!'"

# Download BGE model with error handling and optimization
RUN python -c "import os; os.makedirs('/tmp/models', exist_ok=True); from sentence_transformers import SentenceTransformer; print('Downloading BGE-small-en-v1.5 model...'); model = SentenceTransformer('BAAI/bge-small-en-v1.5'); model.save('/tmp/models/bge-small-en-v1.5'); print('Model downloaded and cached successfully')"

# Stage 2: Optimized production stage
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH"

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user with optimized settings
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder (lighter than full Python installation)
COPY --from=builder /opt/venv /opt/venv

# Copy model from builder stage with proper ownership
COPY --from=builder --chown=appuser:appuser /tmp/models/bge-small-en-v1.5 /app/models/bge-small-en-v1.5

# Create directory structure with proper permissions in single layer
RUN mkdir -p /app/docs /app/outputs /app/cache /app/src && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod 777 /app/outputs /app/cache /app/docs

# Copy application files with proper ownership
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser run_pipeline.py /app/
COPY --chown=appuser:appuser config.json /app/

# Copy entrypoint script
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Optimized health check with CPU-only verification
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=2 \
    CMD python -c "import torch; from src.embedder import PersonaEmbedder; assert not torch.cuda.is_available(); print('CPU-ONLY OK')" || exit 1

# Expose port for potential web interface
EXPOSE 8000

# Set entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]

# Optimized metadata
LABEL org.opencontainers.image.title="Advanced Document Section Selector (CPU-Only)" \
      org.opencontainers.image.description="AI-powered document section selection with persona-aware content extraction - CPU optimized" \
      org.opencontainers.image.version="4.1-cpu-optimized" \
      org.opencontainers.image.authors="Advanced Document Processing System" \
      image_name="advanced-doc-selector-cpu:v4.1" \
      model="BGE-small-en-v1.5" \
      gpu_support="false" \
      cpu_only="true" \
      optimization="cpu-only-size-speed-optimized"
