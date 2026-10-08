import sqlite3

conn = sqlite3.connect("third_eye.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM witness")
rows = cursor.fetchall()

if rows:
    print("\nWitness Records:\n")
    for row in rows:
        print(row)
else:
    print("No witness records found.")

conn.close()