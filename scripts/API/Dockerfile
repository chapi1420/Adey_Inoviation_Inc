# Use an official Python image as the base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask pandas numpy scikit-learn

# Expose API port
EXPOSE 5000

# Run the Fraud Detection API
CMD ["python", "flask_api.py"]
