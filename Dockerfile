# Use a base image with Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install required system packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libxrender-dev \
    libxext6 \
    libsm6 \
    && apt-get clean

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app will run on
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Start the application
CMD ["python", "main.py"]
