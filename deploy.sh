#!/bin/bash

# Exit on error
set -e

# Update system and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-full \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    nginx

# Remove existing venv if exists
rm -rf venv

# Create fresh virtual environment
python3 -m venv venv
source ./venv/bin/activate

# Install Python packages
python3 -m pip install --upgrade pip
python3 -m pip install wheel setuptools
python3 -m pip install flask gunicorn

# Install project requirements
python3 -m pip install -r requirements.txt

# Create gunicorn service
sudo tee /etc/systemd/system/movielens.service << EOL
[Unit]
Description=Gunicorn instance for MovieLens
After=network.target

[Service]
User=root
WorkingDirectory=/root/MovieLensFinalProject
Environment="PATH=/root/MovieLensFinalProject/venv/bin"
ExecStart=/root/MovieLensFinalProject/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 wsgi:app

[Install]
WantedBy=multi-user.target
EOL

# Setup services
sudo systemctl daemon-reload
sudo systemctl start movielens
sudo systemctl enable movielens
sudo systemctl restart nginx

# Show status
sudo systemctl status movielens