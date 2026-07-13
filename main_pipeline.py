"""
Smart City Traffic ANPR — end-to-end pipeline.

Usage:
    python main_pipeline.py --source data/videos/traffic.mp4 --line 0.6 --show

    --source   path to a video file, or 0 for a live webcam
    --line     line position as a fraction of frame height (0.0 top - 1.0 bottom)
    --show     display an annotated preview window while processing
    --save     write the annotated output video to output/results_video.mp4

Output:
    - Rows inserted into database/traffic.db (table vehicle_logs)
    - output/logs.csv (same data, flat file)
    - output/results_video.mp4 (optional annotated video)
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from detector import VehicleDetector          # noqa: E402
from line_counter import LineCounter          # noqa: E402
from plate_ocr import PlateReader             # noqa: E402
import db                                     # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video file path or 0 for webcam")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weight for vehicle detection")
    parser.add_argument("--line", type=float, default=0.6, help="line position, fraction of frame height")
    parser.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    parser.add_argument("--show", action="store_true", help="show a live preview window")
    parser.add_argument("--save", action="store_true", help="save annotated output video")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db.init_db()

    source = 0 if args.source == "0" else args.source
    cap_probe = cv2.VideoCapture(source)
    if not cap_probe.isOpened():
        print(f"ERROR: could not open source '{args.source}'")
        sys.exit(1)
    frame_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25
    cap_probe.release()

    line_y = int(frame_h * args.line)
    counter = LineCounter(line_y=line_y, orientation="horizontal")
    detector = VehicleDetector(model_path=args.model, conf=args.conf)
    plate_reader = PlateReader()

    writer = None
    if args.save:
        out_path = os.path.join(OUTPUT_DIR, "results_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    csv_path = os.path.join(OUTPUT_DIR, "logs.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(["track_id", "vehicle_type", "plate_number", "confidence", "timestamp"])

    print(f"Processing '{args.source}' | frame {frame_w}x{frame_h} | line_y={line_y}")
    t0 = time.time()
    frame_count = 0

    for result in detector.track_stream(source=source, stream=True):
        frame = result.orig_img
        frame_count += 1

        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                vtype = detector.class_name(cls_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{vtype} #{tid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

                crossed = counter.update(tid, (x1, y1, x2, y2), vtype)
                if crossed:
                    vehicle_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    plate_text, conf = plate_reader.read_plate(vehicle_crop)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    db.insert_log(int(tid), vtype, plate_text, conf, ts)
                    csv_writer.writerow([tid, vtype, plate_text or "", f"{conf:.2f}", ts])
                    csv_file.flush()

                    print(f"[CROSSED] id={tid} type={vtype} plate={plate_text} ts={ts}")

        totals = counter.summary()
        y_off = 25
        for vtype, cnt in totals.items():
            cv2.putText(frame, f"{vtype}: {cnt}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_off += 25

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("Smart Traffic ANPR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t0
    print(f"\nDone. {frame_count} frames in {elapsed:.1f}s ({frame_count / max(elapsed, 1e-6):.1f} FPS)")
    print("Final counts:", counter.summary())

    csv_file.close()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
