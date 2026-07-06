"""
Live Object Detection using OpenCV + YOLOv8 (Ultralytics)
-----------------------------------------------------------
Real-time multi-object detection from a webcam feed using YOLOv8
for inference and OpenCV for video capture, drawing, and display.

YOLO detects ALL objects present in a frame in a single pass (it's a
single-shot detector) - so multiple objects (person + chair + laptop +
bottle, etc.) are detected simultaneously, each with its own bounding
box and confidence score.

Default model (yolov8n.pt) is trained on COCO (80 classes) - same
class limitation as before (no "calculator" class exists in COCO),
but YOLO makes it much easier to fine-tune on your OWN custom classes
later if needed (see README).

Usage:
    python live_object_detection_yolo.py

Controls:
    q - quit
    s - save a snapshot of the current frame (saved_frame_<n>.jpg)
"""

import time

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8n.pt"   # nano = fastest; options: yolov8s/m/l/x.pt (bigger = more accurate, slower)
CONF_THRESHOLD = 0.5
CAM_INDEX = 0               # change if you have multiple cameras


def main():
    print(f"Loading {MODEL_NAME} (auto-downloads on first run)...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    prev_time = 0
    snapshot_count = 0

    print("Live object detection running. Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view

        # Run YOLO inference on this frame. verbose=False keeps the console clean.
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # results.boxes contains ALL detected objects in this single frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y1 - 10, text_h + 5)
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - 6),
                (x1 + text_w + 6, label_y + 2),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame, label, (x1 + 3, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        # FPS overlay
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {int(fps)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
        )
        cv2.putText(
            frame, f"Objects: {len(results.boxes)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )

        cv2.imshow("Live Object Detection - YOLOv8 + OpenCV", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_count += 1
            filename = f"saved_frame_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
