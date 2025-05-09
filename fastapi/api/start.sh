#!/bin/bash
cd fastapi/api
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

