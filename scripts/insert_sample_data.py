import sqlite3

data = [
    ("Dr. Prachi Palsodkar", "MH31FE0847", "Teacher"),
    ("Aditya Balpande", "MH31AX9876", "Student"),
    ("Prof. Ankita Tijare", "MH49CD1609", "Teacher")
]

conn = sqlite3.connect('../database/vehicle_records.db')
cursor = conn.cursor()

cursor.executemany('''
INSERT OR IGNORE INTO known_vehicles (name, vehicle_number, person_type)
VALUES (?, ?, ?)
''', data)

conn.commit()
conn.close()

print("✅ Sample data inserted successfully!")
