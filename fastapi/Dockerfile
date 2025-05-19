# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

EXPOSE 8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    ffmpeg \
    ccache \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /app
WORKDIR /app
# Set environment variables
ENV HOME=/home/user
ENV PATH=$PATH:/home/user/.local/bin
ENV PADDLEOCR_HOME=/app/paddleocr_cache
ENV OMP_NUM_THREADS=1

ARG GOOGLE_API_KEY
ARG API_KEYS

ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
ENV API_KEYS=${API_KEYS}

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app
# RUN echo "GOOGLE_API_KEY=${GOOGLE_API_KEY}" > .env

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]