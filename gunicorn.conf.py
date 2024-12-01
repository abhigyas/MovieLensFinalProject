# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:10000"
workers = multiprocessing.cpu_count() * 2 + 1
timeout = 120
keep_alive = 5
max_requests = 1200
max_requests_jitter = 200
worker_class = "sync"