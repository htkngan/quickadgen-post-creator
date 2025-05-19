# Bado-AI

Bado-AI is a powerful advertisement generation platform that leverages Google's Gemini AI models to create compelling ad content and visually appealing images for marketing campaigns. The platform offers APIs to generate advertisement text in Vietnamese and create custom background images with or without product overlays.

## Features

- **Ad Text Generation**: Creates persuasive advertising text in Vietnamese for products and services using Gemini AI.
- **Image Generation**: Produces high-quality advertisement backgrounds using AI image generation.
- **Text Detection and Removal**: Automatically detects and removes text from images using OCR and inpainting technology.
- **Product Image Overlay**: Supports adding product images to generated backgrounds with customizable positioning.
- **REST API Interface**: Provides easy-to-use FastAPI endpoints for all functionality.

## Technology Stack

- **Backend**: FastAPI
- **AI Models**: Google Gemini for text and image generation
- **Image Processing**: PaddleOCR for text detection, LaMa for text removal/inpainting
- **Containerization**: Docker and Docker Compose
- **Language**: Python 3.10

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Google Gemini API Credentials
GOOGLE_API_KEY=your_google_api_key_here

# API Authentication
# Comma-separated list of valid API keys for accessing the endpoints
API_KEYS=key1,key2,key3,key4,key5

# API Configuration
PORT=8000
HOST=0.0.0.0

# Optional parameters
LOG_LEVEL=INFO
```

## Docker Deployment

### Building the Docker Image

Build and start the container using Docker Compose (v2):

```bash
# Build and start containers in detached mode
docker compose -f docker-compose-staging.yml up -d --build
```

### Starting the Application

You can use the provided shell script to build and start the application:

```bash
# Make the script executable
chmod +x run.sh

# Run the deployment script
./run.sh
```

## API Endpoints

- **GET /**: Welcome endpoint
- **GET /health**: Health check endpoint
- **POST /generate-ad**: Generate advertisement text for a product or service
- **POST /generate-image-service**: Generate advertisement content and optionally an image for a service
- **POST /generate-product-ad**: Generate a product advertisement image with optional product overlays

## Local Development

### Prerequisites

- Python 3.10+
- Required system libraries (see Dockerfile)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file
4. Run the application:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Documentation

Once the server is running, access the auto-generated Swagger documentation at:
- http://localhost:8000/docs
- http://localhost:8000/redoc
