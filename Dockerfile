# Use slim python matching project pyc (3.12)
FROM python:3.12-slim
LABEL authors="Cas"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps for video/OpenCV/ffmpeg usage
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project files (including model file yolo11n-pose.pt)
COPY . /app

# Expose port for the app
EXPOSE 80

# Run the main entrypoint
CMD ["python", "main.py"]
