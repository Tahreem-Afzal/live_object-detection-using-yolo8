# Live Object Detection (OpenCV + YOLOv8)

Real-time **multi-object** detection from your webcam using Ultralytics
YOLOv8. All objects present in a frame are detected simultaneously in a
single pass (person, chair, laptop, bottle, book, etc. all at once, each
with its own bounding box).

## How it works

- **OpenCV** handles webcam capture, frame flipping, drawing boxes/labels,
  and the display window.
- **YOLOv8 (nano)** runs the actual detection. It's a single-shot detector:
  one forward pass through the network scans the entire image and outputs
  every object it finds at once — no need to loop or re-run per object.
- The model (`yolov8n.pt`) is trained on **COCO** (80 classes) by default,
  same as before — so it still won't know "calculator" out of the box,
  since that class doesn't exist in COCO. See "Custom classes" below for
  how YOLO makes fixing that far easier than MediaPipe did.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `ultralytics` installs PyTorch as a dependency, so first install is
> heavier (~1-2 GB) and may take a few minutes.

## Run

```bash
python live_object_detection_yolo.py
```

- Press **q** to quit.
- Press **s** to save a snapshot (`saved_frame_N.jpg`).

## Model size options

`yolov8n.pt` (nano) is fastest, used by default. If you have a decent CPU/GPU
and want higher accuracy at the cost of some FPS, swap `MODEL_NAME` to:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| yolov8n.pt | smallest | fastest | lowest |
| yolov8s.pt | small | fast | good |
| yolov8m.pt | medium | moderate | better |
| yolov8l.pt / yolov8x.pt | large | slower | best |

## Custom classes (e.g. calculator, notebook, Pakistani garments)

Unlike MediaPipe's Object Detector, YOLO has a much friendlier fine-tuning
workflow via the same `ultralytics` package:

1. Label your own images (tools like Roboflow or LabelImg export in YOLO format)
2. Organize into a `data.yaml` pointing to train/val image folders + class names
3. Fine-tune:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=50, imgsz=640)
   ```
4. Swap `MODEL_NAME` in the script to your new `best.pt` weights

This is the realistic path if you want GlamourBot-specific detection (e.g.
detecting garment type/category directly from camera, rather than relying
purely on pose landmarks for try-on).

## Troubleshooting

- If webcam doesn't open, change `CAM_INDEX` (0, 1, 2...).
- Lower `CONF_THRESHOLD` (default 0.5) to catch more (noisier) detections.
- First run is slower — it auto-downloads `yolov8n.pt` (~6 MB) and PyTorch
  compiles some kernels on first inference.
