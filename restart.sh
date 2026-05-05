#!/bin/bash
git pull
pkill -f uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
