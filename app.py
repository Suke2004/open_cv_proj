from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import re
from datetime import datetime
import requests
import os

app = Flask(__name__)

# Object Detection Setup
DESIRED_CLASSES = [39, 67, 73, 75]
model = YOLO("yolov8x.pt")
camera_active = False
detect_objects = False

# Date Extraction Setup
DATE_PATTERNS = [
    r'\b\d{1,2}[-/.\s]?\d{1,2}[-/.\s]?\d{2,4}\b',  # DD-MM-YYYY or MM-DD-YYYY
    r'\b\d{4}[-/.\s]?\d{1,2}[-/.\s]?\d{1,2}\b',    # YYYY-MM-DD
    r'\b\d{1,2}[-/.\s]?\d{1,2}\b',                # MM-YY or DD-MM
    r'\b\d{1,2}\s?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s?\d{4}\b',  # DD MMM YYYY
    r'\b\d{1,2}\/\d{1,2}\b',  # MM/DD or M/DD
    r'\b\d{1,2}\/\d{1,2}\/\d{4}\b',  # DD/MM/YYYY
]

def extract_dates(text):
    dates = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            original_date = match.strip()
            formats_to_try = [
                "%d%m%Y", "%m%d%Y", "%Y%m%d", "%m%Y", "%d %b %Y", "%m/%d", "%m/%d/%Y", "%d/%m/%Y"
            ]
            parsed_date = None
            for fmt in formats_to_try:
                try:
                    parsed_date = datetime.strptime(original_date, fmt)
                    break
                except ValueError:
                    continue
            if parsed_date:
                dates.append((original_date, parsed_date))
    return dates

def detect_dates(text):
    dates = extract_dates(text)
    if not dates:
        return "Dates not found"

    # Sort dates by actual date value
    sorted_dates = sorted(dates, key=lambda x: x[1])
    if len(sorted_dates) >= 2:
        manufacturing_date = sorted_dates[0][0]
        expiry_date = sorted_dates[-1][0]
        return {"manufacturing_date": manufacturing_date, "expiry_date": expiry_date}
    else:
        return {"error": "Insufficient date information"}

def extract_text_ocr_space(image_path):
    api_key = 'K83046270988957'  # Replace with your OCR.Space API key
    url = 'https://api.ocr.space/parse/image'

    # Upload and process the image
    with open(image_path, 'rb') as file:
        payload = {
            'apikey': api_key,
            'language': 'eng',
            'OCREngine': 2
        }
        files = {'file': file}
        response = requests.post(url, data=payload, files=files)

    # Parse the response
    result = response.json()
    extracted_text = result.get('ParsedResults', [{}])[0].get('ParsedText', '')
    return extracted_text

def generate_frames():
    global camera_active, detect_objects
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        if detect_objects:
            resized_frame = cv2.resize(frame, (640, 360))
            resized_frame = resized_frame[..., ::-1]
            results = model(resized_frame, conf=0.5, iou=0.4)
            detections = results[0].boxes
            boxes = detections.xyxy.cpu().numpy()
            class_ids = detections.cls.cpu().numpy().astype(int)
            filtered_boxes = [box for box, class_id in zip(boxes, class_ids) if class_id in DESIRED_CLASSES]
            num_objects = len(filtered_boxes)

            for box in filtered_boxes:
                x1, y1, x2, y2 = (box * 2).astype(int)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"Objects Detected: {num_objects}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detect_objects
    detect_objects = True
    return jsonify({"status": "Detection started."})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detect_objects
    detect_objects = False
    return jsonify({"status": "Detection stopped."})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join("uploads", image.filename)
    os.makedirs("uploads", exist_ok=True)
    image.save(image_path)

    extracted_text = extract_text_ocr_space(image_path)
    result = detect_dates(extracted_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
