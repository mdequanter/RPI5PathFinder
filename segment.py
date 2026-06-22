#!/usr/bin/env python3
"""Path segmentation from the Raspberry Pi 5 CSI camera.

Captures frames from the CSI camera (via Picamera2 / libcamera), runs YOLO
segmentation to find the path, computes the heading toward it, and prints
"left" / "right" when a turn is needed. Shows an OpenCV window with the mask
overlay and heading line (works over SSH X forwarding).

Inference takes ~0.5 s/frame, so the loop only processes one frame every
PROCESS_INTERVAL seconds rather than running flat out.
"""

import logging
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc

try:
    from picamera2 import Picamera2
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: picamera2. Install it with: "
        "sudo apt install -y python3-picamera2"
    ) from exc


DETECTION_CONFIDENCE = 0.6
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
TARGET_HEADING = 90.0
HEADING_DEADBAND = 2.0
FRAME_SIZE = (640, 480)
MODEL_PATH = "models/denham.pt"
PROCESS_INTERVAL = 0.5  # seconds between processed frames (~inference time)
WINDOW_NAME = "PathFinder"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("segment")

log.info("Loading model: %s", MODEL_PATH)
model = YOLO(MODEL_PATH, verbose=False)
log.info("Model loaded. Classes: %s", getattr(model, "names", {}))


def get_allowed_mask_indices(result, model_names):
    if result.boxes is None or result.boxes.cls is None:
        return []

    allowed_indices = []
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    for index, class_id in enumerate(class_ids):
        label = str(model_names.get(class_id, "")).strip().lower()
        if label in ALLOWED_PATH_LABELS:
            allowed_indices.append(index)
    return allowed_indices


def compute_heading_to_point(frame, target_x, target_y):
    h, w = frame.shape[:2]
    start_x = w // 2
    start_y = h
    dx = target_x - start_x
    dy = start_y - target_y
    return float(np.degrees(np.arctan2(dy, dx)))


def compute_heading(frame, model):
    """Run segmentation on a frame.

    Returns (heading_degrees, annotated_frame). The annotated frame is a copy
    of the input with the path mask overlay and heading line drawn on it.
    """
    h, w = frame.shape[:2]
    result = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)[0]
    model_names = getattr(model, "names", {})
    vis = frame.copy()
    midpoints = []

    if result.masks is None or len(result.masks.data) == 0:
        return 90.0, vis

    overlay = vis.copy()
    for mask_index in get_allowed_mask_indices(result, model_names):
        if mask_index >= len(result.masks.data):
            continue

        mask = result.masks.data[mask_index].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay[mask > 0] = (0, 255, 0)  # green path overlay

        for row_ratio in SCAN_HEIGHTS:
            y = int(h * row_ratio)
            if y >= h:
                continue
            filled_x = np.where(mask[y, :] > 0)[0]
            if len(filled_x) > 0:
                midpoints.append((int(np.mean(filled_x)), y))

    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

    if not midpoints:
        return 90.0, vis

    for mx, my in midpoints:
        cv2.circle(vis, (mx, my), 3, (0, 0, 255), -1)

    avg_x = int(np.mean([point[0] for point in midpoints]))
    target_y = min(point[1] for point in midpoints)
    cv2.line(vis, (w // 2, h), (avg_x, target_y), (255, 0, 0), 2)
    heading = compute_heading_to_point(frame, avg_x, target_y)
    return heading, vis


def direction_for_heading(heading):
    """Return "left", "right", or None (straight / within deadband).

    Heading is 90 deg straight ahead. A heading below 90 means the path is to
    the right of centre; above 90 means it is to the left.
    """
    error = heading - TARGET_HEADING
    if abs(error) <= HEADING_DEADBAND:
        return None
    return "right" if heading < TARGET_HEADING else "left"


def main():
    log.info("Initialising CSI camera (Picamera2) at %sx%s", *FRAME_SIZE)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    log.info("Camera started. Processing 1 frame every %.2fs "
             "(press 'q' in the window or Ctrl+C to stop).", PROCESS_INTERVAL)

    frame_count = 0
    last_direction = None
    try:
        while True:
            loop_start = time.monotonic()

            # Picamera2 with "RGB888" delivers BGR-ordered data for OpenCV.
            frame = picam2.capture_array()
            frame_count += 1
            if frame_count == 1:
                log.info("First frame read: shape=%s dtype=%s", frame.shape, frame.dtype)

            infer_start = time.monotonic()
            heading, vis = compute_heading(frame, model)
            infer_ms = (time.monotonic() - infer_start) * 1000.0
            direction = direction_for_heading(heading)

            log.debug(
                "frame=%d infer=%.0fms heading=%.1f deg direction=%s",
                frame_count, infer_ms, heading, direction or "straight",
            )

            if direction is not None and direction != last_direction:
                print(direction, flush=True)
            last_direction = direction

            label = f"heading={heading:.1f}  {direction or 'straight'}  {infer_ms:.0f}ms"
            cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, vis)

            # Pace the loop: wait out the rest of the interval (min 1ms so the
            # window repaints), and quit on 'q'.
            elapsed_ms = (time.monotonic() - loop_start) * 1000.0
            wait_ms = max(1, int(PROCESS_INTERVAL * 1000.0 - elapsed_ms))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        log.info("Interrupted. Read %d frames total.", frame_count)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        log.info("Camera stopped.")


if __name__ == "__main__":
    main()
