gunicorn -k uvicorn.workers.UvicornWorker voice2text:app --workers 4 --bind 0.0.0.0:21111

