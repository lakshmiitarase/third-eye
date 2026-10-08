import sqlite3

# Connect to SQLite database
connection = sqlite3.connect("third_eye.db")

# Create cursor
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS investigator(
    investigator_id TEXT PRIMARY KEY,
    agency TEXT NOT NULL,
    password TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS witness(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    witness_name TEXT NOT NULL,
    description TEXT NOT NULL
)
""")

# Save changes
connection.commit()

# Close connection
connection.close()

print("Database created successfully.")