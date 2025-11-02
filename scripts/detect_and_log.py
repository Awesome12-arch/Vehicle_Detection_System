#!/usr/bin/env python3
"""
detect_and_log.py
Integrated live vehicle detection + OCR + color detection + DB logging
Designed for Jetson Orin (GPU) with ultralytics (YOLOv8) + EasyOCR + OpenCV.
"""

import os
import re
import cv2
import time
import sqlite3
import numpy as np
import easyocr
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# -----------------------
# CONFIG - edit if needed
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # project root
IMAGE_SAVE_DIR = BASE_DIR / "output"
DB_PATH = BASE_DIR / "vehicle_records.db"
YOLO_WEIGHTS = "yolov8n.pt"   # will download if missing
CAMERA_INDEX = 0
MIN_OCR_CONF = 0.3            # minimum OCR confidence to accept text items
PLATE_REGEX = re.compile(r'[A-Z0-9]{4,12}')  # rough alphanumeric plate pattern (tweak for country)
COLOR_NAMES = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (200, 30, 30),
    "green": (30, 150, 30),
    "blue": (30, 60, 200),
    "yellow": (220, 200, 30),
    "grey": (128, 128, 128),
    "silver": (190, 190, 190)
}

# -----------------------
# Utilities
# -----------------------
def ensure_dirs():
    IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_number TEXT,
            model TEXT,
            color TEXT,
            person_type TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_record(number, model, color, person_type):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO vehicle_records (vehicle_number, model, color, person_type, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (number, model, color, person_type, timestamp))
    conn.commit()
    conn.close()
    print(f"✅ Record saved: {number}, {model}, {color}, {person_type}")

def map_rgb_to_name(rgb):
    # rgb is (R,G,B)
    r,g,b = rgb
    best = None
    best_dist = float('inf')
    for name, val in COLOR_NAMES.items():
        vr,vg,vb = val
        d = (r-vr)**2 + (g-vg)**2 + (b-vb)**2
        if d < best_dist:
            best_dist = d
            best = name
    return best

def dominant_color_name(bgr_image, k=3):
    # returns human-friendly color name
    img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant = centers[counts.argmax()]
    rgb = tuple(map(int, dominant))
    return map_rgb_to_name(rgb), rgb

def extract_plate_from_roi(roi, reader):
    """
    roi: BGR image (numpy array)
    reader: easyocr.Reader
    returns: best_plate_string or "Unknown"
    """
    # run OCR on ROI
    try:
        results = reader.readtext(roi)
    except Exception as e:
        print("OCR error:", e)
        return "Unknown", []

    candidates = []
    for (bbox, text, conf) in results:
        if conf < MIN_OCR_CONF:
            continue
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(cleaned) >= 4 and PLATE_REGEX.search(cleaned):
            candidates.append((cleaned, conf))

    if not candidates:
        # fallback: try concatenating all text items
        joined = "".join([re.sub(r'[^A-Z0-9]', '', t.upper()) for (_, t, _) in results])
        if len(joined) >= 4:
            return joined, results
        return "Unknown", results

    # choose candidate with highest confidence
    best = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
    return best, results

def classify_person(vehicle_number):
    # Example rule: if starts with MH12 -> Teacher (update as per your DB logic)
    if not vehicle_number or vehicle_number == "Unknown":
        return "Student"
    if vehicle_number.startswith("MH12"):
        return "Teacher"
    # check local list in DB (optional)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, person_type FROM known_vehicles WHERE vehicle_number = ?", (vehicle_number,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[1]
    return "Student"

# -----------------------
# Main runtime
# -----------------------
def main():
    ensure_dirs()
    init_db()

    print("🚀 Starting live detection (press 'q' to quit)...")
    # initialize OCR reader (use GPU if available)
    try:
        reader = easyocr.Reader(['en'], gpu=True)
    except Exception:
        reader = easyocr.Reader(['en'], gpu=False)
    # load YOLO model
    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera not opened - check connection or index.")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed")
            break

        frame_id += 1
        # run YOLO on frame (fast)
        results = model.predict(frame, device=0, conf=0.35, verbose=False)  # returns list-like

        # results[0].boxes contains boxes, results[0].boxes.cls list of classes
        annotated = frame.copy()
        for det in results:
            boxes = det.boxes  # ultralytics Results object
            if boxes is None or len(boxes) == 0:
                continue
            for i,box in enumerate(boxes):
                cls_idx = int(box.cls[0])
                cls_name = det.names[cls_idx] if det.names and cls_idx < len(det.names) else str(cls_idx)
                # only consider common 4-wheeler classes (car, bus, truck)
                if cls_name not in ("car", "truck", "bus", "motorbike", "bicycle"):  # allow broad set
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                # crop ROI - pad a bit
                pad = 8
                h, w = frame.shape[:2]
                xa = max(0, x1 - pad); ya = max(0, y1 - pad)
                xb = min(w, x2 + pad); yb = min(h, y2 + pad)
                roi = frame[ya:yb, xa:xb]

                # OCR attempt on ROI
                plate_text, ocr_results = extract_plate_from_roi(roi, reader)

                # color detection on the ROI
                color_name, rgb = dominant_color_name(roi)

                # vehicle model (approx) -> use class name
                vehicle_model = cls_name

                # classify person
                person_type = classify_person(plate_text)

                # draw
                label = f"{vehicle_model} | {plate_text} | {color_name} | {person_type}"
                cv2.rectangle(annotated, (xa, ya), (xb, yb), (0,255,0), 2)
                cv2.putText(annotated, label, (xa, max(15, ya-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                # save image and insert to DB
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = IMAGE_SAVE_DIR / f"frame_{timestamp}_{frame_id}.jpg"
                cv2.imwrite(str(out_name), annotated)
                insert_record(plate_text, vehicle_model, color_name, person_type)

        # show annotated frame (only if display available)
        try:
            cv2.imshow("VehicleEntry - Live", annotated)
        except Exception:
            # headless: skip show
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
