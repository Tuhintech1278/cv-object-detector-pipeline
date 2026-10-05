"""
Real-time Object Detector & Frame-by-Frame Segmenter Pipeline
-------------------------------------------------------------
Two modes (press 'm' while running to switch):

  contour : classic OpenCV  ->  gray -> Gaussian blur -> adaptive threshold
            -> morphology -> contours -> area ("pixel bounds") filter
            -> shape classification (triangle / square / rectangle / ...)
  yolo    : pre-trained YOLOv8 deep-learning model (detection or segmentation)

Every frame is timed and logged to a CSV file so you can analyse it later
(see analyze.py) and write the performance whitepaper.

Keys:  q = quit   m = switch mode   s = save screenshot
"""

import argparse
import csv
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. CLASSICAL PIPELINE (OpenCV only)
# --------------------------------------------------------------------------
def classify_shape(contour):
    """Name a contour by counting the corners of its simplified outline."""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    corners = len(approx)

    if corners == 3:
        return "triangle"
    if corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        return "square" if 0.90 <= ratio <= 1.10 else "rectangle"
    if corners == 5:
        return "pentagon"

    # many corners: decide between circle and "other" using circularity
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    return "circle" if circularity > 0.80 else "other"


def contour_detect(frame, args):
    """Return (annotated_frame, counts, stage_times_ms)."""
    t = {}

    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)
    t["preprocess_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # local weighted-mean threshold
        cv2.THRESH_BINARY_INV,            # objects white, background black
        args.block, args.c,
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # remove specks
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # close gaps
    t["threshold_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    counts = Counter()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (args.min_area <= area <= args.max_area):   # pixel bounds
            continue
        label = classify_shape(cnt)
        counts[label] += 1
        cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    t["detect_ms"] = (time.perf_counter() - t0) * 1000

    if args.show_mask:
        cv2.imshow("threshold mask", thresh)
    return out, counts, t


# --------------------------------------------------------------------------
# 2. DEEP-LEARNING PIPELINE (YOLO)
# --------------------------------------------------------------------------
_yolo_model = None


def load_yolo(name):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO   # imported here so contour mode works without it
        print(f"[info] loading YOLO model '{name}' (first run downloads it)...")
        _yolo_model = YOLO(name)
    return _yolo_model


def yolo_detect(frame, args):
    model = load_yolo(args.model)
    t = {"preprocess_ms": 0.0, "threshold_ms": 0.0}

    t0 = time.perf_counter()
    result = model.predict(frame, conf=args.conf, verbose=False)[0]
    t["detect_ms"] = (time.perf_counter() - t0) * 1000

    out = result.plot()   # draws boxes (and masks if it is a -seg model)
    counts = Counter(model.names[int(c)] for c in result.boxes.cls)

    # ultralytics also reports its own internal timings (useful for the report)
    t["preprocess_ms"] = result.speed.get("preprocess", 0.0)
    t["detect_ms"] = result.speed.get("inference", t["detect_ms"])
    t["threshold_ms"] = result.speed.get("postprocess", 0.0)   # re-used column: postprocess
    return out, counts, t


# --------------------------------------------------------------------------
# 3. OVERLAY + MAIN LOOP
# --------------------------------------------------------------------------
def draw_overlay(img, counts, fps, mode, frame_idx):
    """Dynamic text panel: live count per class, FPS, mode."""
    lines = [f"Mode: {mode}   FPS: {fps:.1f}   Frame: {frame_idx}",
             f"Total objects: {sum(counts.values())}"]
    lines += [f"{name}: {n}" for name, n in sorted(counts.items())]

    panel_h = 12 + 22 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (300, panel_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)   # semi-transparent box
    for i, text in enumerate(lines):
        cv2.putText(img, text, (12, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def open_source(src):
    if src.isdigit():                                     # webcam index
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows
    else:                                                 # video file path
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[error] could not open source '{src}'")
    return cap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", default="0", help="0 = webcam, or path to a video file")
    p.add_argument("--mode", choices=["contour", "yolo"], default="contour")
    p.add_argument("--model", default="yolov8n.pt",
                   help="yolov8n.pt (detect) or yolov8n-seg.pt (segmentation masks)")
    p.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    p.add_argument("--blur", type=int, default=5, help="Gaussian kernel size (odd number)")
    p.add_argument("--block", type=int, default=21, help="adaptive threshold neighbourhood (odd)")
    p.add_argument("--c", type=int, default=5, help="adaptive threshold constant")
    p.add_argument("--min-area", type=int, default=800, help="smallest contour area in pixels")
    p.add_argument("--max-area", type=int, default=100000, help="largest contour area in pixels")
    p.add_argument("--show-mask", action="store_true", help="also show the binary threshold image")
    p.add_argument("--save-video", action="store_true", help="save annotated output to results/")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--no-display", action="store_true", help="run headless (for benchmarking)")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("results", f"metrics_{stamp}.csv")

    cap = open_source(args.source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(os.path.join("results", f"output_{stamp}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))

    mode = args.mode
    recent = deque(maxlen=30)          # for smooth FPS display
    frame_idx = 0

    with open(csv_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "mode", "preprocess_ms", "threshold_or_post_ms",
                      "detect_ms", "total_ms", "fps", "objects", "counts"])

        print(f"[info] logging metrics to {csv_path}")
        print("[info] press q to quit, m to switch mode, s to save a screenshot")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t_start = time.perf_counter()

            if mode == "contour":
                out, counts, stages = contour_detect(frame, args)
            else:
                out, counts, stages = yolo_detect(frame, args)

            total_ms = (time.perf_counter() - t_start) * 1000
            recent.append(total_ms)
            fps = 1000.0 / (sum(recent) / len(recent))

            log.writerow([frame_idx, mode,
                          round(stages["preprocess_ms"], 3), round(stages["threshold_ms"], 3),
                          round(stages["detect_ms"], 3), round(total_ms, 3), round(fps, 2),
                          sum(counts.values()), dict(counts)])

            out = draw_overlay(out, counts, fps, mode, frame_idx)
            if writer:
                writer.write(out)

            if not args.no_display:
                cv2.imshow("CV Pipeline", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("m"):
                    mode = "yolo" if mode == "contour" else "contour"
                    recent.clear()
                    print(f"[info] switched to {mode} mode")
                elif key == ord("s"):
                    shot = os.path.join("results", f"shot_{frame_idx}.png")
                    cv2.imwrite(shot, out)
                    print(f"[info] saved {shot}")

            if args.max_frames and frame_idx >= args.max_frames:
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[done] processed {frame_idx} frames. Metrics: {csv_path}")


if __name__ == "__main__":
    main()
