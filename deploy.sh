#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install python3-full python3-pip python3-venv nginx -y

# Create and setup virtual environment
python3 -m venv venv
. ./venv/bin/activate

# Install requirements in virtual environment
pip3 install --no-cache-dir -r requirements.txt
pip3 install gunicorn

# Setup Nginx
sudo cp nginx.conf /etc/nginx/sites-available/movielens
sudo ln -s /etc/nginx/sites-available/movielens /etc/nginx/sites-enabled
sudo systemctl restart nginx

# Start application with full path to gunicorn
./venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 wsgi:app