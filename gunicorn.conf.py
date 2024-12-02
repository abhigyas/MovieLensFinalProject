import os
import multiprocessing

# Basic settings
bind = f"0.0.0.0:{int(os.environ.get('PORT', 8000))}"
workers = 1  # Single worker for memory constraints
threads = 2  # Use threads instead of processes
worker_class = "gthread"  # Thread-based workers
worker_connections = 10  # Reduce connections

# Timeouts
timeout = 120  # Increase for long requests
graceful_timeout = 60
keepalive = 5

# Memory optimizations
max_requests = 50
max_requests_jitter = 5
worker_tmp_dir = "/dev/shm"

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Performance tuning
backlog = 100
max_requests_jitter = 5
preload_app = True

# Worker lifecycle
def on_starting(server):
    """Clean up on server start."""
    import gc
    gc.collect()

def pre_fork(server, worker):
    """Pre-fork optimizations."""
    import torch
    torch.cuda.empty_cache()

def post_fork(server, worker):
    """Post-fork cleanup."""
    import gc
    gc.collect()

def worker_abort(worker):
    """Handle worker abort gracefully."""
    import sys
    sys.exit(1)