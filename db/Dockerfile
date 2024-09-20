# Use an official Python 3.10 base image with a slim size
FROM python:3.10-slim

# Set the working directory
WORKDIR /db

# Copy scripted download file
COPY download_file.sh /db

# Download files
RUN echo "Setup dependencies, and instructions to download data ..."
# RUN apk update 
# RUN apk --no-cache add bash curl  
RUN pip install --upgrade pip
RUN pip install gdown
RUN bash download_file.sh 
RUN echo "Done setting up ${PWD}"