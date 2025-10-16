# Use the official Python image as a base
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r /app/requirements.txt

# Copy the application code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Ensure the application runs on Heroku's dynamic port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]