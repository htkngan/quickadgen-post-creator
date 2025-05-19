#!/bin/bash

# Check if the environment parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Environment parameter is required."
    echo "Usage: $0 [staging|uat|pro]"
    exit 1
fi

# Check if the environment parameter is valid
if [ "$1" != "staging" ] && [ "$1" != "uat" ] && [ "$1" != "pro" ]; then
    echo "Error: Invalid environment parameter."
    echo "Valid environments are: staging, uat, pro"
    echo "Usage: $0 [staging|uat|pro]"
    exit 1
fi

# Environment is valid, continue with the script
ENVIRONMENT=$1
echo "Building image for $ENVIRONMENT environment..."

IMAGE_NAME="bado_genai_${ENVIRONMENT}"
sudo docker compose -f docker-compose-${ENVIRONMENT}.yml build

cd ~/bado
sudo docker save -o ${IMAGE_NAME}.tar ${IMAGE_NAME}:$ENVIRONMENT
sudo chown buildsite:buildsite ${IMAGE_NAME}.tar
