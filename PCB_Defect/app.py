import io
import PIL
import cv2
from flask import Flask, request, jsonify, send_file
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model=YOLO ("Model/best.pt")


def size_back(original_size, resized_coordinates, boxes):
    sized_orig_boxes = []
    original_width = original_size[0]
    original_height = original_size[1]
    target_width = resized_coordinates[0]
    target_height = resized_coordinates[1]
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min_resized = int(x_min * original_width / target_width)
        y_min_resized = int(y_min * original_height / target_height)
        x_max_resized = int(x_max * original_width / target_width)
        y_max_resized = int(y_max * original_height / target_height)
        sized_orig_boxes.append((x_min_resized, y_min_resized, x_max_resized, y_max_resized))
    return sized_orig_boxes


# Define the route for image upload and processing
@app.route("/detect_bounding_boxes", methods=["POST"])
def detect_bounding_boxes():
    # Receive the uploaded image
    uploaded_file = request.files["image"]
    
    # Load the image
    image = PIL.Image.open(uploaded_file)
    
    # Resize the image
    resized_coordinates = (640, 640)
    resized_image = image.resize(resized_coordinates, PIL.Image.LANCZOS)

    # Perform inference on the resized image
    results = model(resized_image, save=True)

    # Get the original size of the input image
    original_size = image.size

    # Extract bounding boxes from the inference results
    bounding_boxes = results[0].boxes
    boxes_cv2 = bounding_boxes.xyxy.cpu().numpy()

    # Convert bounding box coordinates back to original size
    original_boxes = size_back(original_size, resized_coordinates, boxes_cv2)
    image_np = np.array(image)

    # Draw bounding boxes on the image
    for box in original_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image_np, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    # Convert the numpy array back to PIL Image
    processed_image = PIL.Image.fromarray(image_np)

    # Convert the processed image to bytes
    img_byte_array = io.BytesIO()
    processed_image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    # Return the processed image as a response
    return send_file(img_byte_array, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
