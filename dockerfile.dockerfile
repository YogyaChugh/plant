# Use an official lightweight Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the model file and application code
COPY model.pth /app/model.pth
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "testingbro.py"]