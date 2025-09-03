# PDF-MCP Server Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Core system tools
    curl \
    wget \
    unzip \
    # PDF processing dependencies
    ghostscript \
    poppler-utils \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-rus \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    # Image processing
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libfreetype6 \
    liblcms2-2 \
    libwebp6 \
    libharfbuzz0b \
    libfribidi0 \
    # Java for Tabula
    default-jre-headless \
    # Graphics libraries
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    # Fonts
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user and directories
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app /app/data /app/logs /app/temp /app/models \
    && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Install the package in development mode
RUN pip install -e .

# Create configuration directory
RUN mkdir -p /app/config && chown appuser:appuser /app/config

# Copy default configuration
COPY --chown=appuser:appuser config/default.yaml /app/config/

# Download and setup models (optional)
RUN mkdir -p /app/models/tesseract /app/models/formula \
    && chown -R appuser:appuser /app/models

# Setup GROBID (optional)
ARG INSTALL_GROBID=false
RUN if [ "$INSTALL_GROBID" = "true" ]; then \
        wget -O /tmp/grobid.zip https://github.com/kermitt2/grobid/archive/0.7.3.zip && \
        unzip /tmp/grobid.zip -d /tmp && \
        mv /tmp/grobid-0.7.3 /app/grobid && \
        chown -R appuser:appuser /app/grobid && \
        rm /tmp/grobid.zip; \
    fi

# Setup formula recognition models (optional)
ARG INSTALL_FORMULA_MODELS=false
RUN if [ "$INSTALL_FORMULA_MODELS" = "true" ]; then \
        python -c "from transformers import AutoTokenizer, AutoModel; \
                   AutoTokenizer.from_pretrained('microsoft/trocr-base-printed'); \
                   AutoModel.from_pretrained('microsoft/trocr-base-printed')" && \
        python -c "import torch; \
                   torch.hub.load('lukas-blecher/LaTeX-OCR', 'model', pretrained=True, trust_repo=True)" || true; \
    fi

# Switch to app user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "pdf_mcp_server.main", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands (can be overridden)
# CMD ["uvicorn", "pdf_mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["gunicorn", "pdf_mcp_server.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Labels for metadata
LABEL maintainer="PDF-MCP Team" \
      version="1.0.0" \
      description="PDF-MCP Server - Lightweight PDF processing with MCP interface" \
      org.opencontainers.image.title="PDF-MCP Server" \
      org.opencontainers.image.description="Lightweight PDF processing server with MCP interface" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="PDF-MCP Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/pdf-mcp/pdf-mcp-server" \
      org.opencontainers.image.documentation="https://github.com/pdf-mcp/pdf-mcp-server/blob/main/README.md"