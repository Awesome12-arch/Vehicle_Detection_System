import os
import cv2
import easyocr
import sqlite3
import numpy as np
from datetime import datetime

# ================================
# Paths and Initialization
# ================================
IMAGE_PATH = r"C:\Projects\VehicleEntrySystem\captured_images\Honda Amaze_Front Side_White.jpg"
OUTPUT_DIR = r"C:\Projects\VehicleEntrySystem\output"
DB_PATH = r"C:\Projects\VehicleEntrySystem\vehicle_records.db"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Starting Vehicle Entry Recognition System...\n")

# ================================
# OCR Function (Improved)
# ================================
def run_ocr(image_path):
    print("🔍 Running OCR on image...")

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR using EasyOCR
    reader = easyocr.Reader(['en'])
    results = reader.readtext(thresh, detail=0)

    # Clean and join text
    if results:
        text = "".join(results)
        text = ''.join(ch for ch in text if ch.isalnum())
    else:
        text = "Unknown"

    print(f"🧾 OCR Detected Number: {text}")
    output_path = os.path.join(OUTPUT_DIR, "ocr_output.jpg")
    cv2.imwrite(output_path, thresh)
    print(f"📁 OCR output saved at: {output_path}")
    return text

# ================================
# Color Detection Function
# ================================
def detect_dominant_color(image_path):
    print("🎨 Detecting dominant color...")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found for color detection")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    img = np.float32(img)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant_color = centers[np.bincount(labels.flatten()).argmax()]
    color_rgb = tuple(map(int, dominant_color))

    color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_patch[:] = color_rgb
    color_path = os.path.join(OUTPUT_DIR, f"color_{color_rgb}.jpg")
    cv2.imwrite(color_path, cv2.cvtColor(color_patch, cv2.COLOR_RGB2BGR))

    print(f"🎨 Detected Color: RGB{color_rgb}")
    print(f"📘 Color patch saved at: {color_path}")
    return f"RGB{color_rgb}"

# ================================
# Vehicle Model Detection (Simple Static)
# ================================
def detect_vehicle_model():
    # For now, we'll just assume it's a car — you can enhance this later
    vehicle_model = "car"
    print(f"🚗 Detected Vehicle Model (approx): {vehicle_model}")
    return vehicle_model

# ================================
# Database Setup
# ================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number TEXT,
            vehicle_model TEXT,
            color TEXT,
            category TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_vehicle_logs():
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
    conn.close()


def save_record(number, vehicle_model, color, category):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO vehicle_records (number, vehicle_model, color, category, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (number, vehicle_model, color, category, timestamp))
    conn.commit()
    conn.close()
    print(f"✅ Record saved: {number}, {vehicle_model}, {color}, {category}\n")

def save_log(number_plate, vehicle_type, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO vehicle_logs (number_plate, vehicle_type, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (number_plate, vehicle_type, confidence, timestamp))
    conn.commit()
    conn.close()
    print(f"📝 Log saved: {number_plate}, {vehicle_type}, {confidence:.2f}")


# ================================
# Main Integration
# ================================
if __name__ == "__main__":
    init_db()
    init_vehicle_logs()

    # Run OCR
    number_plate = run_ocr(IMAGE_PATH)

    # Detect color
    color = detect_dominant_color(IMAGE_PATH)

    # Detect model (static for now)
    vehicle_model = detect_vehicle_model()

    # Categorize based on number pattern (mock logic)
    category = "Teacher" if number_plate.startswith("MH12A") else "Student"

    # Save record
    save_record(number_plate, vehicle_model, color, category)

    print("✅ All processes completed successfully!")

