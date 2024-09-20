#!/bin/sh

pwd
echo "Downloading data ..."
cd ./db
# Assuming download_file.sh is a separate script (modify accordingly)
bash download_file.sh
cd ../

# Run backend
echo "Moving to backend, then run backend"
cd ./backend/app
# Assuming uvicorn requires typing_extensions (modify if not)
if [ -n "$VIRTUAL_ENV" ]; then
  # Use uvicorn within virtual environment (if activated)
  uvicorn main:app --host=0.0.0.0 --port=8000 --reload
else
  # Use uvicorn outside virtual environment
  python -m uvicorn main:app --host=0.0.0.0 --port=8000 --reload
fi