FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT