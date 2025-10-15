# Use a stable Python version that has prebuilt wheels
FROM python:3.10-slim

# Install minimal system deps needed by pillow, torchvision, etc.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend only (we build backend service inside /app)
COPY backend/ ./backend/

# Ensure pip/setuptools/wheel are recent
RUN python -m pip install --upgrade pip setuptools wheel

# Install backend requirements first (excluding torch/torchvision)
RUN pip install -r backend/requirements.txt

# Install CPU-only PyTorch wheels (explicit)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy product images (if any) already included in backend folder (they were committed)
# (backend/ already copied above)
# Expose the port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
