FROM python:3.11.3-slim

RUN mkdir /app

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Copy the agents directory and Python files
COPY agents/ agents/
COPY agent_controller.py agent_controller.py
COPY main.py main.py

# Create a directory for temporary file uploads
RUN mkdir -p /tmp/uploads

# Set environment variables for file handling
ENV TEMP_UPLOAD_DIR=/tmp/uploads

ENTRYPOINT [ "python", "main.py"]

