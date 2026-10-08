# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Install with extra index url for CPU-only torch to keep image size small
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["python", "-m", "src.api.main"]
