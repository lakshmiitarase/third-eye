import sqlite3

conn = sqlite3.connect("third_eye.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM investigator")

rows = cursor.fetchall()

if rows:
    print("Registered Investigators:\n")
    for row in rows:
        print(row)
else:
    print("No records found.")

conn.close()