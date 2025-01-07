#!/bin/bash

# Exit on error
set -e

# Update system and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3-full python3-pip python3-venv python3-distutils nginx -y

# Remove existing venv if exists
rm -rf venv

# Create fresh virtual environment
python3 -m venv venv
source ./venv/bin/activate

# Install core dependencies first
pip3 install --upgrade pip
pip3 install setuptools wheel
pip3 install flask gunicorn

# Now install other requirements
pip3 install --no-cache-dir -r requirements.txt

# Create gunicorn service file
sudo tee /etc/systemd/system/movielens.service << EOL
[Unit]
Description=Gunicorn instance to serve MovieLens
After=network.target

[Service]
User=root
WorkingDirectory=/root/MovieLensFinalProject
Environment="PATH=/root/MovieLensFinalProject/venv/bin"
ExecStart=/root/MovieLensFinalProject/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 wsgi:app

[Install]
WantedBy=multi-user.target
EOL

# Setup and start services
sudo systemctl daemon-reload
sudo systemctl start movielens
sudo systemctl enable movielens

# Setup Nginx
sudo cp nginx.conf /etc/nginx/sites-available/movielens
sudo ln -sf /etc/nginx/sites-available/movielens /etc/nginx/sites-enabled
sudo systemctl restart nginx

# Show status
sudo systemctl status movielens