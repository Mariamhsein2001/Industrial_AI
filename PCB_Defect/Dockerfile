FROM ultralytics/ultralytics:latest-python


# Set the working directory in the container
WORKDIR /app
# Copy the application code into the container
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt


# Run app.py when the container launches
CMD ["python", "app.py"]

