from flask import Flask, request, jsonify, make_response
import face_recognition
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from mtcnn import MTCNN
import dlib
import logging
import json
import io
import threading
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Initialize MTCNN and Dlib CNN detectors
mtcnn_detector = MTCNN()
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Resize settings
IMAGE_SIZE = (1024, 1024)


def resize_and_normalize_image(image):
    """
    Resize and normalize the image for consistent processing.
    """
    if image.size[0] > IMAGE_SIZE[0] or image.size[1] > IMAGE_SIZE[1]:
        resized_image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    else:
        resized_image = image  # Skip resizing for smaller images
    normalized_image = ImageOps.equalize(resized_image)
    return normalized_image


def detect_faces_with_mtcnn(image):
    """
    Use MTCNN to detect faces in an image and return the bounding boxes.
    """
    detections = mtcnn_detector.detect_faces(image)
    face_locations = []
    for detection in detections:
        if detection['confidence'] >= 0.90:  # Adjusted confidence threshold
            x, y, width, height = detection['box']
            top, right, bottom, left = y, x + width, y + height, x
            face_locations.append((top, right, bottom, left))
        else:
            logging.info(f"Filtered out low-confidence detection: {detection['confidence']}")
    return face_locations


def detect_faces_with_dlib(image, timeout=5):
    """
    Use Dlib's CNN face detector to detect faces in an image with a timeout (Windows-compatible).
    """
    def run_detection():
        try:
            detections.extend(cnn_face_detector(image, 1))
        except Exception as e:
            logging.error(f"Dlib detection error: {e}")

    detections = []
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.start()
    detection_thread.join(timeout)

    if detection_thread.is_alive():
        logging.error("Dlib face detection timed out")
        return []

    face_locations = []
    for detection in detections:
        rect = detection.rect
        face_locations.append((rect.top(), rect.right(), rect.bottom(), rect.left()))
    return face_locations


@app.route('/encode', methods=['POST'])
def encode_faces():
    """
    Endpoint to encode faces and return encodings along with bounding boxes.
    """
    try:
        if 'image' not in request.files:
            logging.error("No image file provided.")
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        logging.info(f"Received image file: {image_file.filename}")

        image = Image.open(image_file).convert("RGB")
        processed_image = resize_and_normalize_image(image)
        np_image = np.array(processed_image)

        # Step 1: Detect faces with MTCNN
        face_locations = detect_faces_with_mtcnn(np_image)
        logging.info(f"MTCNN detected {len(face_locations)} face(s).")

        # Step 2: If MTCNN fails, use Dlib CNN detector
        if not face_locations:
            face_locations = detect_faces_with_dlib(np_image)
            logging.info(f"Dlib CNN detected {len(face_locations)} face(s).")

        # Step 3: No faces detected
        if not face_locations:
            logging.warning("No faces detected in the image.")
            return jsonify({'error': 'No faces detected in the image'}), 400

        # Step 4: Generate face encodings
        encodings = face_recognition.face_encodings(np_image, face_locations)
        logging.info(f"Generated {len(encodings)} encoding(s).")

        if not encodings:
            logging.warning(
                f"Face(s) detected at {face_locations}, but no encodings generated. "
                "Possible reasons: blurry image, low resolution, or partial faces."
            )
            return jsonify({'error': 'Face detected, but no encodings generated'}), 400

        return jsonify({
            'encodings': [list(encoding) for encoding in encodings],
            'face_locations': face_locations,
            'num_faces': len(encodings)
        }), 200

    except Exception as e:
        logging.error(f"Exception during face encoding: {str(e)}")
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 500


@app.route('/draw', methods=['POST'])
def draw_bounding_boxes():
    """
    Endpoint to draw bounding boxes with names on an image and return the processed image.
    """
    try:
        if 'image' not in request.files or 'bounding_boxes' not in request.form or 'names' not in request.form:
            return jsonify({'error': 'Image, bounding boxes, and names are required'}), 400

        # Retrieve the image
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Retrieve the bounding boxes and names
        bounding_boxes = json.loads(request.form['bounding_boxes'])
        names = json.loads(request.form['names'])

        # Validation
        if not isinstance(bounding_boxes, list) or not all(
            isinstance(b, list) and len(b) == 4 for b in bounding_boxes
        ) or not isinstance(names, list) or len(names) != len(bounding_boxes):
            return jsonify({'error': 'Invalid bounding boxes or names'}), 400

        # Draw bounding boxes
        for (top, right, bottom, left), name in zip(bounding_boxes, names):
            draw.rectangle([left, top, right, bottom], outline="blue", width=3)
            draw.text((left, top - 10), name, fill="white")

        # Save the result to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'image/jpeg'
        return response

    except Exception as e:
        logging.error(f"Exception occurred while drawing bounding boxes: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
