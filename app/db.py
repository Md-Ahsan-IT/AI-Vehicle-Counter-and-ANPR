"""
Small SQLite helper used by both the pipeline and the Streamlit dashboard.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "traffic.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "schema.sql")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def insert_log(track_id, vehicle_type, plate_number, plate_confidence, timestamp):
    conn = get_connection()
    conn.execute(
        """INSERT INTO vehicle_logs (track_id, vehicle_type, plate_number, plate_confidence, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (track_id, vehicle_type, plate_number, plate_confidence, timestamp),
    )
    conn.commit()
    conn.close()


def fetch_all_logs():
    conn = get_connection()
    cur = conn.execute("SELECT * FROM vehicle_logs ORDER BY id DESC")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    return cols, rows


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {os.path.abspath(DB_PATH)}")
