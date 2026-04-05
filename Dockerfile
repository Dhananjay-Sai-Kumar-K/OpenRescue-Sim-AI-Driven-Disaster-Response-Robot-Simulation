# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port HF Spaces uses (7860 by default) or specify differently
EXPOSE 7860

# Command to run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
