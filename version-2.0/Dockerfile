# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies + missing ones found in code
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mysql-connector-python plotly matplotlib

# Pre-download EasyOCR models (English) to avoid runtime downloads
RUN python -c "import easyocr; easyocr.Reader(['en'])"

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to ensure container is healthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "ui_app.py"]
