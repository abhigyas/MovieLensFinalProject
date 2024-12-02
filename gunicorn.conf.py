import multiprocessing

bind = "0.0.0.0:10000"
# Reduce number of workers
workers = 2
# Reduce timeout
timeout = 60
# Add max requests settings
max_requests = 1000
max_requests_jitter = 50
# Add worker class
worker_class = "sync"
# Add worker connections
worker_connections = 1000
# Add memory settings
worker_tmp_dir = "/dev/shm"