"""
Optional FastAPI service — exposes the same data as the Streamlit dashboard
over a REST API, useful if another system (e.g. a city traffic-control app)
needs to consume the logs programmatically.

Run with:
    uvicorn app.api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""
import os
import sys

from fastapi import FastAPI

sys.path.append(os.path.dirname(__file__))
import db  # noqa: E402

app = FastAPI(title="Smart Traffic ANPR API")
db.init_db()


@app.get("/")
def root():
    return {"status": "ok", "service": "smart-traffic-anpr"}


@app.get("/logs")
def get_logs(limit: int = 100):
    cols, rows = db.fetch_all_logs()
    rows = rows[:limit]
    return [dict(zip(cols, row)) for row in rows]


@app.get("/counts")
def get_counts():
    cols, rows = db.fetch_all_logs()
    counts = {}
    type_idx = cols.index("vehicle_type")
    for row in rows:
        vtype = row[type_idx]
        counts[vtype] = counts.get(vtype, 0) + 1
    return counts
