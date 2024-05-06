import os
import cv2
import requests
import numpy as np
from flask import Flask, request, send_file, jsonify, abort, render_template, redirect, url_for
import io

app = Flask(__name__)

# Route to handle image processing business logic
@app.route("/defect", methods=["POST"])
def detect_defects():
    try:
        # Receive image from the request
        image_data = request.files["image"].read()

        # Send the image to the defect detection service
        defect_detection_response = send_image_for_defect_detection(image_data)

        # Check if the request was successful
        if defect_detection_response.status_code != 200:
            # If there was an error in the request, abort with an error response
            abort(defect_detection_response.status_code)

        # Extract the image from the response
        defect_image_bytes = defect_detection_response.content

        # Return the image as a response
        return send_file(io.BytesIO(defect_image_bytes), mimetype='image/png')

    except Exception as e:
        # Log the error
        app.logger.error(f"An error occurred: {e}")

        # Return an error response
        return jsonify({"error": "An error occurred while processing the request"}), 500

# Route to handle prediction of failure based on JSON data
@app.route("/failure", methods=["POST"])
def predict_failure():
    try:
        # Receive JSON data from the request
        data = request.json

        # Send the JSON data to the predict failure service
        predict_failure_response = send_data_for_predict_failure(data)

        # Check if the request was successful
        if predict_failure_response.status_code != 200:
            # If there was an error in the request, abort with an error response
            abort(predict_failure_response.status_code)

        # Extract the prediction from the response
        prediction = predict_failure_response.json()

        # Return the prediction as a response
        return jsonify(prediction)

    except Exception as e:
        # Log the error
        app.logger.error(f"An error occurred: {e}")

        # Return an error response
        return jsonify({"error": "An error occurred while processing the request"}), 500

# Function to send image to defect detection service
def send_image_for_defect_detection(image_data):
    return requests.post(
        os.getenv("DEFECT_DETECTION_URL"), 
        files={"image": image_data}
    )

# Function to send JSON data to predict failure service
def send_data_for_predict_failure(data):
    return requests.post(
        os.getenv("PREDICT_FAILURE_URL"), 
        json=data
    )

# Route to render image upload form
@app.route("/upload", methods=["GET"])
def upload_image():
    return render_template("upload.html")

# Default route to redirect to the image upload form
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("upload_image"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT"))
