"""YOLOv8 Real-time Webcam Object Detection with PolyInfer.

This example demonstrates:
1. Exporting YOLOv8 to ONNX
2. Running real-time inference with the fastest available backend
3. Drawing bounding boxes and FPS overlay

Requirements:
    pip install ultralytics opencv-python

Run: python examples/yolov8_webcam.py
     python examples/yolov8_webcam.py --backend onnxruntime
     python examples/yolov8_webcam.py --device cuda
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import polyinfer as pi

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Generate random colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def export_yolov8_onnx(model_name: str = "yolov8n", imgsz: int = 640) -> str:
    """Export YOLOv8 model to ONNX format."""
    output_dir = Path(f"./models/{model_name}-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    if onnx_path.exists():
        print(f"ONNX model exists: {onnx_path}")
        return str(onnx_path)

    print(f"Exporting {model_name} to ONNX...")
    try:
        from ultralytics import YOLO

        model = YOLO(f"{model_name}.pt")
        model.export(format="onnx", imgsz=imgsz, simplify=True)
        # Move exported file to models directory
        exported = Path(f"{model_name}.onnx")
        if exported.exists():
            exported.rename(onnx_path)
        print(f"Exported: {onnx_path}")
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        raise

    return str(onnx_path)


def preprocess(frame: np.ndarray, input_size: int = 640) -> tuple:
    """Preprocess frame for YOLOv8.

    Returns:
        - Preprocessed tensor (1, 3, H, W)
        - Scale factor
        - Padding (pad_x, pad_y)
        - Original size (H, W)
    """
    orig_h, orig_w = frame.shape[:2]

    # Calculate scale and new dimensions
    scale = min(input_size / orig_h, input_size / orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square (letterbox)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # Convert BGR -> RGB, normalize, transpose to NCHW
    tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]

    return tensor, scale, (pad_x, pad_y), (orig_h, orig_w)


def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple,
    orig_size: tuple,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> list:
    """Post-process YOLOv8 output.

    Args:
        output: Model output (1, 84, 8400)
        scale: Preprocessing scale factor
        pad: Padding (pad_x, pad_y)
        orig_size: Original image size (H, W)
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold for NMS

    Returns:
        List of detections: [{"bbox": (x1,y1,x2,y2), "class_id": int, "confidence": float}, ...]
    """
    # Transpose: (1, 84, 8400) -> (8400, 84)
    preds = output[0].T

    # Split boxes and class scores
    boxes = preds[:, :4]  # cx, cy, w, h
    scores = preds[:, 4:]  # 80 class scores

    # Get best class per detection
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    # Filter by confidence
    mask = confidences > conf_thresh
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes) == 0:
        return []

    # Convert to corner format
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Remove padding and scale to original size
    pad_x, pad_y = pad
    orig_h, orig_w = orig_size
    x1 = np.clip((x1 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_y) / scale, 0, orig_h)

    # NMS
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(), conf_thresh, iou_thresh)

    if len(indices) == 0:
        return []

    indices = indices.flatten()
    detections = []
    for i in indices:
        detections.append({
            "bbox": (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
            "class_id": int(class_ids[i]),
            "class_name": COCO_CLASSES[class_ids[i]],
            "confidence": float(confidences[i]),
        })

    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        color = tuple(int(c) for c in COLORS[class_id])

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam Detection with PolyInfer")
    parser.add_argument("--model", default="yolov8n", help="YOLOv8 model variant")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, directml")
    parser.add_argument("--backend", default=None, help="Backend: onnxruntime, openvino, tensorrt")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 Webcam Detection - PolyInfer")
    print("=" * 60)
    print()

    # Export model
    onnx_path = export_yolov8_onnx(args.model)

    # Load model with PolyInfer
    print(f"\nLoading model with device={args.device}, backend={args.backend}...")
    model = pi.load(onnx_path, device=args.device, backend=args.backend)
    print(f"Loaded: {model}")
    print()

    # Warmup
    print("Warming up...")
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(5):
        model(dummy)

    # Quick benchmark
    results = model.benchmark(dummy, warmup=5, iterations=20)
    print(f"Benchmark: {results['mean_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    print()

    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\nControls: Q=Quit, S=Screenshot")
    print()

    frame_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            start = time.perf_counter()
            tensor, scale, pad, orig_size = preprocess(frame)
            output = model(tensor)
            detections = postprocess(output, scale, pad, orig_size)
            elapsed = (time.perf_counter() - start) * 1000

            # Track FPS
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_ms = np.mean(frame_times)
            fps = 1000 / avg_ms

            # Draw results
            frame = draw_detections(frame, detections)

            # Draw overlay
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} ({avg_ms:.1f}ms) | {model.backend_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLOv8 - PolyInfer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    main()
