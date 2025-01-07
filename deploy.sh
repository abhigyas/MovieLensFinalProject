#!/bin/bash

# Update system and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3-full python3-pip python3-venv python3-distutils nginx -y

# Remove existing venv if exists
rm -rf venv

# Create fresh virtual environment
python3 -m venv venv
source ./venv/bin/activate

# Upgrade pip and install setuptools first
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel

# Install requirements
pip3 install --no-cache-dir -r requirements.txt
pip3 install gunicorn

# Setup Nginx
sudo cp nginx.conf /etc/nginx/sites-available/movielens
sudo ln -s /etc/nginx/sites-available/movielens /etc/nginx/sites-enabled
sudo systemctl restart nginx

# Start application
./venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 wsgi:app