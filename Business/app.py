# import os
# import cv2
# import requests
# import numpy as np
# from flask import Flask, request, send_file, jsonify, abort,render_template
# import io

# app = Flask(__name__)

# @app.route("/business", methods=["POST"])
# def business():
#     try:
#         # Receive image from the request
#         image_data = request.files["image"].read()

#         # Send the image to the defect detection service
#         defect_detection_response = requests.post(
#             os.getenv("Defect_Detection_URL"), files={"image": image_data}
#         )

#         # Check if the request was successful
#         if defect_detection_response.status_code != 200:
#             # If there was an error in the request, abort with an error response
#             abort(defect_detection_response.status_code)

#         # Extract the image from the response
#         image_bytes = defect_detection_response.content

#         # Return the image as a response
#         return send_file(io.BytesIO(image_bytes), mimetype='image/png')

#     except Exception as eo:
#         # Log the error
#         app.logger.error(f"An error occurred: {e}")

#         # Return an error response
#         return jsonify({"error": "An error occurred while processing the request"}), 500
# @app.route("/upload", methods=["GET"])
# def upload_image():
#     return render_template("upload.html")
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=os.getenv("PORT"))
import os
import cv2
import requests
import numpy as np
from flask import Flask, request, send_file, jsonify, abort, render_template, redirect, url_for
import io

app = Flask(__name__)

# Route to handle image processing business logic
@app.route("/business", methods=["POST"])
def business():
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
        image_bytes = defect_detection_response.content

        # Return the image as a response
        return send_file(io.BytesIO(image_bytes), mimetype='image/png')

    except Exception as e:
        # Log the error
        app.logger.error(f"An error occurred: {e}")

        # Return an error response
        return jsonify({"error": "An error occurred while processing the request"}), 500

# Function to send image to defect detection service
def send_image_for_defect_detection(image_data):
    return requests.post(
        os.getenv("Defect_Detection_URL"), 
        files={"image": image_data}
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
