import io
import PIL
from PIL import ImageFont
from flask import Flask, request, send_file
import numpy as np
from ultralytics import YOLO
import logging
import os
app = Flask(__name__)

model=YOLO ("Model/best.pt")

def convert_coord(original_size, resized_coordinates, boxes):
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

def draw_bounding_boxes(image, original_boxes, class_names):

    """Draw bounding boxes and class names on the input image."""
    # processed_image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image)
    for idx, box in enumerate(original_boxes):
        x_min, y_min, x_max, y_max = box
        class_name = class_names[idx]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        font = ImageFont.truetype('arial.ttf' ,size=35)
        draw.text((x_min, y_min - 35), class_name, fill="red",font=font)
    return image

def process_image(uploaded_file,output_size=(1000, 600)):
    """Process the uploaded image and return the processed image as bytes."""
    # Load the image
    image = PIL.Image.open(uploaded_file)

    # Resize the image
    resized_coordinates = (640, 640)
    resized_image = image.resize(resized_coordinates, PIL.Image.LANCZOS)

    # Perform inference on the resized image
    results = model(resized_image)

    # Get the original size of the input image
    original_size = image.size

    # Extract bounding boxes from the inference results
    bounding_boxes = results[0].boxes
        # Extract bounding boxes from the inference results
    bounding_boxes = results[0].boxes
    if not bounding_boxes:
        # Create a new image object to draw the OK text
        processed_image = image.copy()
        draw = PIL.ImageDraw.Draw(processed_image)
        font = ImageFont.truetype('arial.ttf' ,size=35)
        draw.rectangle([0, 0, original_size[0], original_size[1]], outline="green", width=2)
        draw.text((10, 10), "OK", fill="green",font=font)
    else:
        # Convert bounding box coordinates back to original size
        boxes_cv2 = bounding_boxes.xyxy.cpu().numpy()
        original_boxes = convert_coord(original_size, resized_coordinates, boxes_cv2)
        class_mapping = results[0].names
        class_names = [class_mapping.get(int(cls.item()), "Unknown") for cls in bounding_boxes.cls]
        processed_image = draw_bounding_boxes(image, original_boxes, class_names)
    # processed_image = processed_image.resize(output_size, PIL.Image.BOX)
    img_byte_array = io.BytesIO()
    processed_image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    return img_byte_array

@app.route("/detect_defects", methods=["POST"])
def detect_defects():
    try:
        uploaded_file = request.files["image"]
        processed_image = process_image(uploaded_file)
        return send_file(processed_image, mimetype='image/png')
    except Exception as e:
        # Log the error
        logging.error(f"Error processing image: {e}")
        # Return an error response
        return "Error processing image", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0",  port=os.getenv("PORT"))
