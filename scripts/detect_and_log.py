import cv2
import torch
import easyocr
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO

# === Paths ===
OUTPUT_DIR = "/home/jetson-orin/Vehicle_Detection_System/output"
DB_PATH = "/home/jetson-orin/Vehicle_Detection_System/vehicle_records.db"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Additional database setup for vehicle_records ===
def init_vehicle_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number TEXT,
            vehicle_model TEXT,
            color TEXT,
            category TEXT,
<<<<<<< HEAD
=======
            image_path TEXT,
>>>>>>> 2c42e26 (Updated Jetson deployment - detect_and_log integration and database setup)
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

<<<<<<< HEAD
def save_vehicle_record(number, vehicle_model, color, category):
=======

def save_vehicle_record(number, vehicle_model, color, category, image_path):
>>>>>>> 2c42e26 (Updated Jetson deployment - detect_and_log integration and database setup)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
<<<<<<< HEAD
        INSERT INTO vehicle_records (number, vehicle_model, color, category, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (number, vehicle_model, color, category, timestamp))
    conn.commit()
    conn.close()
    print(f"✅ Record saved to vehicle_records: {number}, {vehicle_model}, {color}, {category}")
=======
        INSERT INTO vehicle_records (number, vehicle_model, color, category, image_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (number, vehicle_model, color, category, image_path, timestamp))
    conn.commit()
    conn.close()
    print(f"✅ Record saved: {number}, {vehicle_model}, {color}, {category}, {os.path.basename(image_path)}")

>>>>>>> 2c42e26 (Updated Jetson deployment - detect_and_log integration and database setup)

# === Initialize model, OCR, and DB ===
print("🚀 Initializing system...")

model = YOLO("yolov8n.pt")
print("✅ YOLOv8 model loaded successfully")

reader = easyocr.Reader(['en'])
print("✅ EasyOCR initialized")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicle_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    number_plate TEXT,
    vehicle_type TEXT,
    confidence REAL,
    timestamp TEXT
)
""")
conn.commit()
print("✅ Database connected successfully")

# Initialize vehicle_records table
init_vehicle_records()

# === Use camera or test image ===
USE_IMAGE = False  # 🔁 Change to True if you want to test a static image

if USE_IMAGE:
    IMAGE_PATH = "/home/jetson-orin/Vehicle_Detection_System/captured_images/Honda_Amaze_Front_Side_White.jpg"
    frame = cv2.imread(IMAGE_PATH)
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access camera")

print("🚗 Starting vehicle detection... (Press Ctrl+C to stop)")

try:
    while True:
        if USE_IMAGE:
            frames = [frame]
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not captured. Skipping...")
                continue
            frames = [frame]

        for f in frames:
            results = model(f, conf=0.5)

            for result in results:
                boxes = result.boxes
                annotated_frame = result.plot()

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    # Extract region of interest (ROI)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = f[y1:y2, x1:x2]

                    # OCR attempt
                    number_plate = None
                    if roi.size > 0:
                        ocr_result = reader.readtext(roi)
                        if ocr_result:
                            number_plate = ocr_result[0][-2].replace(" ", "")

                    if number_plate:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cursor.execute(
                            "INSERT INTO vehicle_logs (number_plate, vehicle_type, confidence, timestamp) VALUES (?, ?, ?, ?)",
                            (number_plate, label, conf, timestamp)
                        )
                        conn.commit()

                        print(f"✅ Detected: {number_plate} | {label} | {conf:.2f} | {timestamp}")

                        # --- Save detailed record automatically ---
                        vehicle_model = label
                        color = "Unknown"  # optional — can be upgraded later
                        category = "Teacher" if number_plate.startswith("MH12A") else "Student"
<<<<<<< HEAD
                        save_vehicle_record(number_plate, vehicle_model, color, category)
=======
                        save_vehicle_record(number_plate, vehicle_model, color, category, save_path)
>>>>>>> 2c42e26 (Updated Jetson deployment - detect_and_log integration and database setup)

                # Save annotated output
                save_path = os.path.join(OUTPUT_DIR, f"detected_{datetime.now().strftime('%H%M%S')}.jpg")
                cv2.imwrite(save_path, annotated_frame)
                print(f"🖼️ Saved detection to {save_path}")

        if USE_IMAGE:
            break

except KeyboardInterrupt:
    print("\n🛑 Detection stopped manually.")

finally:
    if not USE_IMAGE:
        cap.release()
    conn.close()
    print("✅ Resources released. Goodbye!")
<<<<<<< HEAD
=======

>>>>>>> 2c42e26 (Updated Jetson deployment - detect_and_log integration and database setup)
