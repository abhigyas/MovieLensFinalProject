import os

# Fix port binding
bind = f"0.0.0.0:{int(os.environ.get('PORT', 8000))}"

# Reduce workers for memory
workers = 1

# Increase timeout
timeout = 300

# Reduce requests
max_requests = 100
max_requests_jitter = 10

# Basic worker settings
worker_class = "sync"
worker_connections = 100

# Memory settings
worker_tmp_dir = "/dev/shm"

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'

# Preload app
preload_app = True

# Add graceful timeout
graceful_timeout = 120

# Force port in app
raw_env = [f"PORT={os.environ.get('PORT', 8000)}"]