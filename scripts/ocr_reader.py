import cv2
import easyocr
from pathlib import Path
import os
import sqlite3
from datetime import datetime

# ✅ Step 1: Define input and output paths
image_path = r"C:\Projects\VehicleEntrySystem\captured_images\Honda Amaze_Front Side_White.jpg"
output_dir = Path(r"C:\Projects\VehicleEntrySystem\output")  # your folder
db_path = r"C:\Projects\VehicleEntrySystem\database\vehicle_records.db"

# Create output and database directories if they don't exist
output_dir.mkdir(parents=True, exist_ok=True)
Path(db_path).parent.mkdir(parents=True, exist_ok=True)

# ✅ Step 2: Load image
image = cv2.imread(image_path)
if image is None:
    print("❌ Error: Could not load image. Check the file path.")
    exit()

# ✅ Step 3: Initialize OCR reader
reader = easyocr.Reader(['en'])
results = reader.readtext(image)

# ✅ Step 4: Extract and display detected text
detected_text = ""
for (bbox, text, prob) in results:
    detected_text += text + " "
    # Draw rectangle around detected area
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# ✅ Step 5: Generate timestamped output file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_image_path = output_dir / f"{base_name}_{timestamp}_ocr.png"
output_text_path = output_dir / f"{base_name}_{timestamp}_ocr.txt"

# ✅ Step 6: Save results (image + text)
cv2.imwrite(str(output_image_path), image)
with open(output_text_path, "w") as f:
    f.write(detected_text.strip())

print("\n🧾 OCR Results:")
if detected_text.strip():
    print(f"Detected text: {detected_text.strip()}")
else:
    print("⚠️ No text detected.")

print(f"📸 Annotated image saved to: {output_image_path}")
print(f"📝 OCR text saved to: {output_text_path}")

# ✅ Step 7: Store OCR data into SQLite database
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captured_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_number TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    if detected_text.strip():
        cursor.execute('INSERT INTO captured_vehicles (vehicle_number) VALUES (?)', (detected_text.strip(),))
        conn.commit()
        print("💾 Data inserted into database successfully!")
    else:
        print("⚠️ No data inserted (empty OCR result).")

except Exception as e:
    print(f"❌ Database error: {e}")

finally:
    conn.close()

