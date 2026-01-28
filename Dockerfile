FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     gcc     curl     && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create a non-root user
RUN useradd -m -u 1000 eee && chown -R eee:eee /app
USER eee

# Expose port 8080 for Streamlit
EXPOSE 8080

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3   CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Start Streamlit on port 8080
CMD ["streamlit", "run", "ui/app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
