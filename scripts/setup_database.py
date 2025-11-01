import sqlite3

# Connect to database (creates it if not exists)
conn = sqlite3.connect('../database/vehicle_records.db')
cursor = conn.cursor()

# Create main table
cursor.execute('''
CREATE TABLE IF NOT EXISTS vehicle_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    vehicle_number TEXT,
    vehicle_model TEXT,
    vehicle_color TEXT,
    person_type TEXT,
    timestamp TEXT
)
''')

# Create reference table (for known teachers/students)
cursor.execute('''
CREATE TABLE IF NOT EXISTS known_vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    vehicle_number TEXT UNIQUE,
    person_type TEXT
)
''')

conn.commit()
conn.close()

print("✅ Database setup completed successfully!")
