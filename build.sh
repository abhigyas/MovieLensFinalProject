#!/bin/bash
echo "Starting build process..."

# Update pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Build completed"