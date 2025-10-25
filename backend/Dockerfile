# Multi-stage build for MSME Chatbot Backend

# Stage 1: Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser msme_chatbot_agents/ ./msme_chatbot_agents/

# Install custom agents package
RUN pip install --no-cache-dir --user -e ./msme_chatbot_agents/dist/msme_chatbot_agents-1.0.0-py3-none-any.whl || \
    pip install --no-cache-dir --user ./msme_chatbot_agents/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
##HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
   ## CMD python -c "import requests; requests.get('http://localhost:8000/docs')" || exit 1

# Set working directory to backend
WORKDIR /app/backend

# Run the application
CMD ["bash", "-lc" ,"uvicorn main:app --host 0.0.0.0 --port ${PORT}" ]
