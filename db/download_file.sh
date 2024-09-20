#!/bin/sh

# Download and extract dataset
echo "Start downloading and extracting files .... "

# Download dataset from Google Drive
gdown --id 18Fx1dmi_OJVdGE5ObU8NR08z7DNjCd9n --output dataset.zip

# Check if download was successful
if [ $? -eq 0 ]; then
  # Extract dataset
  unzip dataset.zip
  rm dataset.zip
  echo "Done!"
else
  echo "Download failed!"
  exit 1
fi