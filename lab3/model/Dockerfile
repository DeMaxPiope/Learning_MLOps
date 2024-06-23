# Using the official Python image
FROM python:3.8-slim

# Setting the working directory in the container
WORKDIR /model

# Copying files
COPY . .

# Dependency installation
RUN pip install --no-cache-dir -r requirements.txt

# Launching the application
CMD ["python", "model.py"]
