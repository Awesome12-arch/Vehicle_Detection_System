import sqlite3
from tabulate import tabulate

# Path to your database
db_path = r"C:\Projects\VehicleEntrySystem\database\vehicle_records.db"

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch all records
cursor.execute("SELECT * FROM captured_vehicles ORDER BY id DESC")
rows = cursor.fetchall()

# Display results
if rows:
    print("\n🚗 Vehicle Entry Records:\n")
    print(tabulate(rows, headers=["ID", "Vehicle Number", "Timestamp"], tablefmt="grid"))
else:
    print("⚠️ No records found in the database.")

conn.close()
