import cv2
import face_recognition
import pickle
import numpy as np
import os
from ultralytics import YOLO

# Load YOLOv8 model for object detection
yolo_model = YOLO("yolo11n.pt")  # Ensure this file is in your directory or use "yolov8s.pt" for better accuracy

# Load the trained face classifier
classifier_path = "classifier.pkl"
if not os.path.exists(classifier_path):
    print("Classifier file not found. Train the model first!")
    exit()

with open(classifier_path, "rb") as f:
    knn, labels = pickle.load(f)

# Start webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Run YOLO object detection
    yolo_results = yolo_model(frame)

    # Predict faces
    predictions = []
    if face_encodings:
        closest_distances = knn.kneighbors(face_encodings, n_neighbors=1)
        is_recognized = [dist[0] <= 0.5 for dist in closest_distances[0]]

        predictions = [
            (pred, loc) if recognized else ("UNKNOWN", loc)
            for pred, loc, recognized in zip(knn.predict(face_encodings), face_locations, is_recognized)
        ]

    # Draw YOLO object detection results
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = result.names[cls]  # Object name
            confidence = float(box.conf[0])

            # Draw bounding box for objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw results for face recognition
    for name, (top, right, bottom, left) in predictions:
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('YOLO + Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
