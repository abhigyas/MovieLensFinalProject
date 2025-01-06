
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-dev nginx -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Setup Gunicorn
sudo cp nginx.conf /etc/nginx/sites-available/movielens
sudo ln -s /etc/nginx/sites-available/movielens /etc/nginx/sites-enabled
sudo systemctl restart nginx

# Start application
gunicorn --workers 3 --bind 127.0.0.1:8000 wsgi:app