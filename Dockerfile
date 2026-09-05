FROM python:3.11-slim


# -------------------------------------------------
# Python settings
# -------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app


# -------------------------------------------------
# Project directory
# -------------------------------------------------

WORKDIR /app


# -------------------------------------------------
# Linux dependencies
# -------------------------------------------------

RUN apt-get update && \
    apt-get install -y \
        --no-install-recommends \
        libgomp1 \
        libmagic1 \
        curl && \
    rm -rf /var/lib/apt/lists/*


# -------------------------------------------------
# Python dependencies
# -------------------------------------------------

COPY requirements.txt .


# CPU PyTorch
RUN pip install --no-cache-dir \
    --timeout 300 \
    --retries 10 \
    torch==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir \
    -r requirements.txt


# -------------------------------------------------
# Application
# -------------------------------------------------

COPY . .


# -------------------------------------------------
# FastAPI
# -------------------------------------------------

EXPOSE 8000


CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]