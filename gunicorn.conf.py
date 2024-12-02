import os
import multiprocessing
import gc

# Basic settings
bind = f"0.0.0.0:{int(os.environ.get('PORT', 8000))}"
workers = 1  # Single worker for memory constraints
threads = 2  # Use threads instead of processes
worker_class = "gthread"  # Thread-based workers
worker_connections = 10  # Reduce connections

# Timeouts
timeout = 30  # Reduce timeout
graceful_timeout = 30
keepalive = 2

# Memory optimizations
max_requests = 50
max_requests_jitter = 5
worker_tmp_dir = "/dev/shm"

# Lifecycle hooks
def on_starting(server):
    """Clean up on server start."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

def pre_fork(server, worker):
    """Pre-fork optimizations."""
    gc.collect()

def post_fork(server, worker):
    """Post-fork cleanup."""
    gc.collect()

def pre_request(worker, req):
    """Pre-request cleanup."""
    gc.collect()

def post_request(worker, req, environ, resp):
    """Post-request cleanup."""
    gc.collect()

def worker_exit(server, worker):
    """Clean up on worker exit."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

def worker_abort(worker):
    """Handle worker abort gracefully."""
    gc.collect()
    import sys
    sys.exit(1)

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Performance tuning
backlog = 100
max_requests_jitter = 5
preload_app = True