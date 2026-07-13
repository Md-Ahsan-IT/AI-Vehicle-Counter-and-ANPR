# Smart City Traffic — Vehicle Tracking & License Plate Recognition (ANPR)

End-to-end pipeline: video → YOLOv8 vehicle detection → ByteTrack tracking →
virtual line-crossing counter → license-plate crop → EasyOCR → SQLite → 
Streamlit dashboard / FastAPI.

```
Video Input → Frame Extraction (OpenCV) → YOLOv8 Detection → Tracking
→ Line Crossing → Plate Crop → EasyOCR → SQLite → Streamlit / FastAPI
```

## 1. Requirements

- Python 3.9–3.12
- ~3 GB free disk (model weights + EasyOCR language models download on first run)
- Works on CPU; a GPU (CUDA) makes it much faster but is not required

## 2. Install

```bash
cd smart-traffic-anpr
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The YOLOv8 vehicle-detection weight (`yolov8n.pt`) downloads automatically
the first time you run the pipeline — no manual step needed.
EasyOCR also downloads its English recognition model automatically on first use.

## 3. Get a test video

The project needs a traffic video to run on. Options:

**A) Use your own CCTV / dashcam clip** — just drop any `.mp4` into
`data/videos/`.

**B) Use the sample videos linked in the original brief:**
Google Drive folder:
`https://drive.google.com/drive/folders/11n5u1B3BAppISJl7OgLNII3shY9FnLsV`
Open that link in your browser, download a clip, and save it to
`data/videos/traffic.mp4`. (I can't download from Google Drive myself — it
needs a browser login/consent step — so this one step has to be done on
your machine.)

**C) Free stock traffic footage** (no login needed) — search "free traffic
CCTV footage" on Pexels or Pixabay, or use any short traffic clip you have.

## 4. (Optional but recommended) Train a real plate detector

Out of the box, the pipeline finds plates with a simple heuristic crop
(lower-center part of each vehicle box) — it works, but a trained detector
is much more accurate.

```bash
# 1. Go to the dataset page and download in "YOLOv8" format:
#    https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
#    (needs a free Roboflow account) → unzip into data/plate_dataset/

# 2. Train (50 epochs is a reasonable starting point, ~30-60 min on CPU,
#    a few minutes on GPU):
python train_plate_model.py --data data/plate_dataset/data.yaml --epochs 50
```

This writes `models/plate_model.pt`, which `main_pipeline.py` picks up
automatically on the next run. Skip this step and the fallback crop is used
instead — no code changes needed either way.

## 5. Run the detection pipeline

```bash
python main_pipeline.py --source data/videos/traffic.mp4 --line 0.6 --save --show
```

- `--line 0.6` → counting line drawn at 60% down the frame (tune per video)
- `--show`     → live preview window (press `q` to quit early)
- `--save`     → writes annotated video to `output/results_video.mp4`
- `--source 0` → use a live webcam instead of a file

Every time a tracked vehicle crosses the line, its type + cropped plate OCR
result + timestamp are written to `database/traffic.db` and appended to
`output/logs.csv`.

## 6. View the dashboard

In a **second terminal** (leave the pipeline running in the first if you
want it to update live):

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501` — shows live counts, a class-wise bar
chart, a traffic-over-time line chart, and the full plate log table.

## 7. (Optional) REST API

```bash
uvicorn app.api:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`, endpoints: `GET /logs`, `GET /counts`.

## 8. Project structure

```
smart-traffic-anpr/
├── data/
│   ├── videos/            ← put traffic.mp4 here
│   └── images/
├── src/
│   ├── detector.py         YOLOv8 detection + tracking
│   ├── line_counter.py     virtual line-crossing logic
│   └── plate_ocr.py        plate crop + EasyOCR
├── models/
│   └── plate_model.pt      (optional, from train_plate_model.py)
├── database/
│   ├── schema.sql
│   └── traffic.db          created automatically on first run
├── app/
│   ├── db.py
│   ├── streamlit_app.py    dashboard
│   └── api.py              optional FastAPI service
├── output/
│   ├── results_video.mp4
│   └── logs.csv
├── main_pipeline.py         run this first
├── train_plate_model.py     optional plate-detector training
└── requirements.txt
```

## 9. Troubleshooting

- **Slow on CPU** — normal for `yolov8n.pt` + EasyOCR without a GPU; use a
  shorter test clip first, or add `--conf 0.5` to reduce false detections.
- **`cv2.imshow` errors on a headless server** — drop `--show`, keep
  `--save`, and check `output/results_video.mp4` afterward instead.
- **EasyOCR download fails** — it needs internet on first run only, to
  fetch its recognition model; after that it's cached locally.
- **Plate text is empty/wrong often** — expected with the heuristic crop;
  do step 4 (train a real plate detector) for a big accuracy jump.
