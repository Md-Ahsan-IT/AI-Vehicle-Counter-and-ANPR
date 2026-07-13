-- Smart Traffic ANPR - SQLite schema
CREATE TABLE IF NOT EXISTS vehicle_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id        INTEGER NOT NULL,
    vehicle_type    TEXT NOT NULL,
    plate_number    TEXT,
    plate_confidence REAL,
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vehicle_type ON vehicle_logs (vehicle_type);
CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_logs (timestamp);
