"""
OPTIONAL: train a dedicated license-plate detector.

Without this, main_pipeline.py still works — it falls back to a heuristic
crop of the vehicle's lower-center region before running OCR. Training this
model makes plate localisation (and therefore OCR accuracy) much better.

Steps:
1. Download the dataset in YOLOv8 format from Roboflow:
   https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
   -> "Download Dataset" -> format "YOLOv8" -> unzip into data/plate_dataset/
   (the folder must contain data.yaml, train/, valid/, test/)

2. Run:
   python train_plate_model.py --data data/plate_dataset/data.yaml --epochs 50

3. The best weights get copied to models/plate_model.pt automatically,
   which main_pipeline.py / src/plate_ocr.py will then pick up.
"""
import argparse
import shutil
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data.yaml from the Roboflow export")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base", default="yolov8n.pt", help="base weight to fine-tune from")
    args = parser.parse_args()

    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="output",
        name="plate_training",
    )

    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    dest = os.path.join(os.path.dirname(__file__), "models", "plate_model.pt")
    shutil.copy(best_weights, dest)
    print(f"Trained plate model copied to {dest}")


if __name__ == "__main__":
    main()
