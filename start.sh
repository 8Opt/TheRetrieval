#!/bin/sh

pwd
echo "Downloading data ..."
cd ./db
# Assuming download_file.sh is a separate script (modify accordingly)
bash download_file.sh
cd ../

# Run backend
echo "Moving to backend and install requirements"
cd ./backend
python3 -m pip install -r requirements.txt
cd ../

# Install PyTorch
if ! python3 -c "import torch"; then
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
 
# Moving to app and run the app
echo "Moving to app and run the app"
cd ./backend/app
# Assuming uvicorn requires typing_extensions (modify if not)
if [ -n "$VIRTUAL_ENV" ]; then
  # Use uvicorn within virtual environment (if activated)
  uvicorn main:app --host=0.0.0.0 --port=8000 --reload
else
  # Use uvicorn outside virtual environment
  python -m uvicorn main:app --host=0.0.0.0 --port=8000 --reload
fi