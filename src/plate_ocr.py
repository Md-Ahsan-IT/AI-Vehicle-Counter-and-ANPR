"""
License plate localisation + OCR.

Two modes:
1. If a trained plate-detector weight exists at models/plate_model.pt
   (see train_plate_model.py) it is used to find the exact plate box
   inside the cropped vehicle image -> much more accurate.
2. Otherwise falls back to a heuristic crop (lower-center portion of the
   vehicle bounding box, which is usually where the plate sits) so the
   pipeline still runs end-to-end without a custom-trained model.

OCR is done with EasyOCR (no external binary needed, unlike Tesseract).
"""
import os
import re
import cv2
import numpy as np
import easyocr

_PLATE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "plate_model.pt")

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        # english only, set gpu=True if you have CUDA set up
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _heuristic_plate_crop(vehicle_img: np.ndarray) -> np.ndarray:
    h, w = vehicle_img.shape[:2]
    y1 = int(h * 0.55)
    y2 = h
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    return vehicle_img[y1:y2, x1:x2]


def _clean_plate_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\-]", "", text)
    return text


class PlateReader:
    def __init__(self):
        self.plate_model = None
        if os.path.exists(_PLATE_MODEL_PATH):
            from ultralytics import YOLO
            self.plate_model = YOLO(_PLATE_MODEL_PATH)

    def read_plate(self, vehicle_img: np.ndarray):
        """
        vehicle_img: BGR crop of the *vehicle* (not the plate) as produced
                     by the detector's bounding box.
        Returns (plate_text, confidence) or (None, 0.0) if nothing readable.
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None, 0.0

        plate_crop = None
        if self.plate_model is not None:
            res = self.plate_model.predict(vehicle_img, verbose=False, conf=0.4)[0]
            if len(res.boxes) > 0:
                # take the highest-confidence plate box
                best = max(res.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].tolist()]
                plate_crop = vehicle_img[y1:y2, x1:x2]

        if plate_crop is None or plate_crop.size == 0:
            plate_crop = _heuristic_plate_crop(vehicle_img)

        if plate_crop.size == 0:
            return None, 0.0

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        reader = _get_reader()
        results = reader.readtext(gray)
        if not results:
            return None, 0.0

        # concatenate all detected text fragments, weighted by confidence
        text = "".join([r[1] for r in results])
        conf = float(np.mean([r[2] for r in results]))
        text = _clean_plate_text(text)

        return (text if text else None), conf
