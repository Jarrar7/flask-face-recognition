# Use an official lightweight Python image
FROM python:3.10

# Install system dependencies required by dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5001 (or your Flask port)
EXPOSE 5001

# Command to start Flask
CMD ["gunicorn", "-b", "0.0.0.0:5001", "face_service:app"]
