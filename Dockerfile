# Use official Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY ./src ./src
COPY ./models ./models
COPY ./data ./data

# Expose port FastAPI will run on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
