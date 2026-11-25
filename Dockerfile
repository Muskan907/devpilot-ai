# Multi-Agent Autonomous Software Engineering System
# Container image for Hugging Face Spaces (Docker SDK)
# Spaces runs containers as UID 1000, so the app user must be UID 1000.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    HOME=/home/user

# Install system dependencies (git is required by the sandbox / Git tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY --chown=user:user requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=user:user . .

# Ensure runtime directories exist and are writable by the app user
RUN mkdir -p /app/sandbox_workspace/runs /app/sandbox_workspace/sample_repo /home/user/.streamlit && \
    chown -R user:user /app /home/user

USER user

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
