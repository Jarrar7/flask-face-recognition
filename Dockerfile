# Use an official lightweight Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5001 (or your Flask port)
EXPOSE 5001

# Command to start Flask
CMD ["gunicorn", "-b", "0.0.0.0:5001", "face_service:app"]
