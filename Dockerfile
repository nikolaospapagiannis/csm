FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and models if they don't exist
RUN mkdir -p data models

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV HOST=0.0.0.0
ENV PORT=5000
ENV DEBUG=False
ENV SPEAKER_ID=0
ENV MAX_AUDIO_LENGTH=5000
ENV CHUNK_SIZE=60

# Expose port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["python"]

# Set default command
CMD ["app.py"]