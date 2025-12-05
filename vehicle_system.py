import cv2
import torch
import csv
import os
import easyocr
import math
from datetime import datetime
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------- SOURCE SELECTION -----------------
source = input("Enter video file path (press Enter for webcam): ").strip()

if source == "":
    # Webcam: change index if needed (0 = default laptop cam, 1 = external)
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam index {cam_index}")
    # Optional: set resolution for live cam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    # Video file
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video file not found: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {source}")
# ----------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("🚀 Running on:", device)

# Create folder to save detected vehicles
os.makedirs("detected_vehicles", exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Vehicle classes to detect
target_classes = ['car', 'bus', 'motorcycle', 'van']

# EasyOCR reader (CPU)
reader = easyocr.Reader(['en'])

# Deep SORT tracker
tracker = DeepSort(max_age=30)

# Track history for speed calculation
track_history = defaultdict(list)

# CSV logging
csv_file = open('vehicle_detections.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'Timestamp', 'Frame', 'TrackID', 'Class', 'Confidence',
    'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height', 'Area',
    'Center_X', 'Center_Y', 'Speed (px/sec)', 'NumberPlate'
])

cv2.namedWindow("Vehicle Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Tracker", 1280, 720)

frame_count = 0

while True:
    start_time = datetime.now()
    ret, frame = cap.read()

    if not ret:
        print("✅ End of stream or camera error.")
        break

    # Resize for faster inference (optional)
    frame = cv2.resize(frame, (640, 360))

    frame_count += 1
    timestamp = datetime.now()

    # YOLOv5 inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Prepare detections for Deep SORT
    track_inputs = []
    for _, row in detections.iterrows():
        label = row['name']
        if label in target_classes:
            track_inputs.append([
                [int(row['xmin']), int(row['ymin']),
                 int(row['xmax']), int(row['ymax'])],
                float(row['confidence']),
                label
            ])

    # Update tracks
    tracks = tracker.update_tracks(track_inputs, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        label = track.get_det_class()
        conf = 0.85  # placeholder

        # Boundaries check
        h, w, _ = frame.shape
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ID:{track_id}', (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Center and size
        center_x, center_y = (l + r) // 2, (t + b) // 2
        width, height = r - l, b - t
        area = width * height

        # Speed calculation (px/sec)
        track_history[track_id].append((timestamp, center_x, center_y))
        speed = 0
        if len(track_history[track_id]) >= 2:
            t1, x1, y1 = track_history[track_id][-2]
            t2, x2, y2 = track_history[track_id][-1]
            dt = (t2 - t1).total_seconds()
            dist = math.hypot(x2 - x1, y2 - y1)
            speed = dist / dt if dt > 0 else 0

        # Crop and save vehicle
        vehicle_crop = frame[t:b, l:r]
        image_name = f"vehicle_{track_id}_{timestamp.strftime('%H%M%S')}.jpg"
        if vehicle_crop.size > 0:
            cv2.imwrite(f"detected_vehicles/{image_name}", vehicle_crop)

        # OCR every 10 frames
        plate_number = "Skipped"
        if frame_count % 10 == 0 and vehicle_crop.size > 0:
            try:
                plate_text = reader.readtext(vehicle_crop, detail=0)
                plate_number = plate_text[0] if plate_text else "N/A"
            except Exception as e:
                print(f"OCR failed for TrackID {track_id}: {e}")
                plate_number = "OCR_Error"

        # Log to CSV
        csv_writer.writerow([
            timestamp.strftime('%Y-%m-%d %H:%M:%S'), frame_count,
            track_id, label, conf,
            l, t, r, b,
            width, height, area,
            center_x, center_y,
            round(speed, 2), plate_number
        ])

    # FPS display
    fps = 1.0 / (datetime.now() - start_time).total_seconds()
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, "Vehicle Detection + Tracking", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow('Vehicle Tracker', frame)

    # Close on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
csv_file.close()
