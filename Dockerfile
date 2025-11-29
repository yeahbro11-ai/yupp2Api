# Use the official Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (uncomment if needed)
# RUN apt-get update && apt-get install -y \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY yyapi.py .
COPY model.py .

# Create directory for model configuration files
RUN mkdir -p /app/model

# Expose port (configurable via environment variable)
EXPOSE ${PORT:-8001}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBUG_MODE=false

# Health check (port configured via environment variable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/models || exit 1

# Start command
CMD ["python", "-c", "from yyapi import main; main()"]
