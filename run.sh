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
    echo "Usage: $0 [staging|uat|pro] optional: build"
    echo "Example: $0 staging build --> start with building"
    echo "Example: $0 staging --> start without building"
    exit 1
fi

sudo docker compose -f docker-compose-$1.yml down

if [ "$2" == "build" ]; then
    sudo docker compose -f docker-compose-$1.yml up -d --build
else
    sudo docker compose -f docker-compose-$1.yml up -d
fi
