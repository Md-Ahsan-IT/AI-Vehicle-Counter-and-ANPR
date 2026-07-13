"""
Smart Traffic ANPR — Streamlit dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))
import db  # noqa: E402

st.set_page_config(page_title="Smart Traffic ANPR Dashboard", layout="wide")

db.init_db()

st.title("🚦 Smart City Traffic — ANPR Dashboard")
st.caption("Live vehicle counts, license-plate logs and traffic analytics")

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5s", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Run `python main_pipeline.py --source <video> --save` in another "
                  "terminal to populate this dashboard live.")


def load_data():
    cols, rows = db.fetch_all_logs()
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total vehicles logged", len(df))
if not df.empty:
    col2.metric("Cars", int((df["vehicle_type"] == "car").sum()))
    col3.metric("Bikes", int((df["vehicle_type"] == "bike").sum()))
    col4.metric("Trucks + Buses", int(df["vehicle_type"].isin(["truck", "bus"]).sum()))
else:
    col2.metric("Cars", 0)
    col3.metric("Bikes", 0)
    col4.metric("Trucks + Buses", 0)

st.subheader("Class-wise counts")
if not df.empty:
    counts = df["vehicle_type"].value_counts()
    st.bar_chart(counts)
else:
    st.info("No detections logged yet. Start the pipeline to see data here.")

st.subheader("Traffic over time")
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    per_minute = df.set_index("timestamp").resample("1min").size()
    st.line_chart(per_minute)

st.subheader("License plate logs")
st.dataframe(df, use_container_width=True, height=400)

if auto_refresh:
    time.sleep(5)
    st.rerun()
