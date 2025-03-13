# Use an official Python runtime as a parent image
FROM python:3.11

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the essential files
COPY requirements.txt .
COPY app.py .
COPY resnet50_mps.pth .


# Make port 8080 available to the world outside this container
EXPOSE 8080

# Create script to run either Flask or Jupyter
RUN echo '#!/bin/bash\nif [ "$1" = "flask" ]; then\n  python app.py\nelse\n  jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root --no-browser\nfi' > /app/start.sh && chmod +x /app/start.sh

# Set the default command
ENTRYPOINT ["/app/start.sh"]
CMD ["flask"] 