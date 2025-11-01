import sqlite3
from datetime import datetime

# Replace this with your detected OCR result dynamically later
vehicle_number = "MH31FE0847"
vehicle_model = "Honda Amaze"
vehicle_color = "White"

conn = sqlite3.connect('../database/vehicle_records.db')
cursor = conn.cursor()

# Check if known
cursor.execute("SELECT name, person_type FROM known_vehicles WHERE vehicle_number = ?", (vehicle_number,))
result = cursor.fetchone()

if result:
    name, person_type = result
else:
    name, person_type = "Unknown", "Student"

# Log entry
cursor.execute('''
INSERT INTO vehicle_entries (name, vehicle_number, vehicle_model, vehicle_color, person_type, timestamp)
VALUES (?, ?, ?, ?, ?, ?)
''', (name, vehicle_number, vehicle_model, vehicle_color, person_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

conn.commit()
conn.close()

print(f"✅ Vehicle {vehicle_number} logged successfully ({person_type})")
