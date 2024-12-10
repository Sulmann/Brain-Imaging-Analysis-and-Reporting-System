from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import shutil
import numpy as np
import tensorflow as tf
import cv2
import os

from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as PDFImage


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from your React frontend

# Load your classification model
CLASSIFICATION_MODEL_PATH = "vgg16_brain_tumor_model1.h5"
classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)


# Load your YOLO model
YOLO_MODEL_PATH = "/home/sulman/Downloads/weights/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# Classes for classification
CLASS_NAMES = ['No Tumor', 'Tumor Detected']

# Define a fixed folder for YOLO results
YOLO_RESULTS_FOLDER = "runs/detect/predict"  # Fixed folder for YOLO results
#os.makedirs(YOLO_RESULTS_FOLDER, exist_ok=True)  # Ensure the folder exists

# Function to preprocess image for classification
def preprocess_image(image, target_size=(512, 512)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] != 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    return np.expand_dims(image_array, axis=0)

# Function to generate PDF report
def create_pdf_report(prediction, filename, bounding_boxes=None, class_ids=None, quadrants=None, output_image_path=None):
    pdf_path = os.path.join("uploads", f"{filename.split('.')[0]}_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Add header
    c.setFont("Helvetica-Bold", 18)  # Larger font for header
    c.drawString(160, 780, "Brain Tumor Detection and Reporting System")
    c.line(50, 770, 550, 770)  # Line below the header

    # Write details below the header
    c.setFont("Helvetica-Bold", 14)  # Bold and larger font for subheadings
    c.drawString(50, 740, "Tumor Detection Report")

    c.setFont("Helvetica", 12)  # Normal font for details
    c.drawString(50, 710, f"Filename: {filename}")
    c.drawString(50, 690, f"Prediction: {prediction}")
    
    # Add bounding box and quadrant details if tumor is detected
    if prediction == "Tumor Detected" and bounding_boxes is not None:
        c.setFont("Helvetica-Bold", 14)  # Bold font for "Tumor Localization"
        c.drawString(50, 660, "Tumor Localization:")

        c.setFont("Helvetica", 12)  # Normal font for bounding box details
        for i, bbox in enumerate(bounding_boxes):
            # Format YOLO bounding box coordinates and corresponding quadrant
            bbox_str = (
                f"Box {i + 1}: (x1: {bbox[0]:.2f}, y1: {bbox[1]:.2f}, "
                f"x2: {bbox[2]:.2f}, y2: {bbox[3]:.2f}), Quadrant: {quadrants[i]}"
            )
            c.drawString(70, 640 - i * 20, bbox_str)

    # Add the YOLO output image at the bottom of the page
    try:
        if output_image_path and os.path.exists(output_image_path):
            c.drawString(200, 300, "YOLO Output Image:")
            c.drawImage(output_image_path, 80, 80, width=400, height=400)  # Adjust size and position
    except Exception as e:
        print(f"Error adding image to PDF: {e}")

    # Add footer
    c.line(50, 40, 550, 40)  # Line above the footer
    c.setFont("Helvetica-Oblique", 10)  # Italic font for footer
    c.drawString(200, 20, "FYP Project - FAST University Peshawar Campus")

    c.save()
    return pdf_path

# Function to predict tumor using classification model
def classify_tumor(image_path):
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    prediction = classification_model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return CLASS_NAMES[predicted_class]

# Function to determine the quadrant of a tumor
def get_quadrant(image_width, image_height, bbox):
    """
    Determine the quadrant of a tumor based on its bounding box center.
    Quadrants:
    "Top-left", "Top-right", "Bottom-left", "Bottom-right"
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if center_x < image_width / 2 and center_y < image_height / 2:
        return "Top-left"
    elif center_x >= image_width / 2 and center_y < image_height / 2:
        return "Top-right"
    elif center_x < image_width / 2 and center_y >= image_height / 2:
        return "Bottom-left"
    elif center_x >= image_width / 2 and center_y >= image_height / 2:
        return "Bottom-right"

# Function to localize tumor using YOLO
def localize_tumor(image_path):
    # Clear existing files in the YOLO results folder
    predict_folder = os.path.join(YOLO_RESULTS_FOLDER, "predict")
    if os.path.exists(predict_folder):
        shutil.rmtree(predict_folder)
    
    # Run YOLO inference on the image and save to the fixed folder
    results = yolo_model.predict(
        source=image_path,
        save=True,
        imgsz=512,
        conf=0.3,
        project=YOLO_RESULTS_FOLDER,
        name="predict",
        exist_ok=True
    )
    
    # Extract saved YOLO output image path
    output_image_files = os.listdir(predict_folder)
    if output_image_files:
        output_image_path = os.path.join(predict_folder, output_image_files[0])
    else:
        output_image_path = None

    # Read the image to get its dimensions
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Extract bounding boxes and class ids
    bounding_boxes = results[0].boxes.xyxy.numpy()  # Extract bounding box coordinates
    class_ids = results[0].boxes.cls.numpy()  # Extract class IDs

    # Calculate quadrants for each bounding box
    quadrants = [
        get_quadrant(image_width, image_height, bbox) for bbox in bounding_boxes
    ]

    return bounding_boxes, class_ids, quadrants, output_image_path

# Update the /predict route to include quadrant information and use the fixed YOLO results folder
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(image_path)

        # Classification
        prediction = classify_tumor(image_path)

        # If "Tumor Detected", add bounding boxes, quadrants, and YOLO results path
        bounding_boxes = class_ids = quadrants = None
        output_image_path = None
        if prediction == "Tumor Detected":
            bounding_boxes, class_ids, quadrants, output_image_path = localize_tumor(image_path)
            bounding_boxes = bounding_boxes.tolist()
            class_ids = class_ids.tolist()

        # Create PDF report
        pdf_report_path = create_pdf_report(
            prediction,
            file.filename,
            bounding_boxes=bounding_boxes,
            class_ids=class_ids,
            quadrants=quadrants,  # Added quadrants
            output_image_path=output_image_path  # Pass YOLO output image path
        )

        # Response payload
        response = {
            'prediction': prediction,
            'pdf_report': f"http://localhost:5001/download/{file.filename}",
            'bounding_boxes': bounding_boxes,
            'quadrants': quadrants,  # Descriptive quadrant names
            'class_ids': class_ids,
            'yolo_results_path': YOLO_RESULTS_FOLDER  # Path of the fixed YOLO results folder
        }

        return jsonify(response)

    return jsonify({'error': 'Invalid file format'}), 400

# Route for downloading the PDF report
@app.route('/download/<filename>', methods=['GET'])
def download_report(filename):
    pdf_path = os.path.join("uploads", f"{filename.split('.')[0]}_report.pdf")
    return send_file(pdf_path, as_attachment=True)

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

