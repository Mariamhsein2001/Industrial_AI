from ultralytics import YOLO
import os
from flask import Flask, request, send_file

app = Flask(__name__)

model = YOLO('Model/best.pt')

@app.route("/detect_defect", methods=["POST"])
def detect_defect():
    # Receive image from the request
    image_data = request.files["image"].read()
    results = model(image_data)
    
    # Save the output image
    output_path = "example_output.jpg"
    results.save(output_path)
    
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
