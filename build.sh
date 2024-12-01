#!/bin/bash
echo "Starting build process..."
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Build completed"
