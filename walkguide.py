#!/usr/bin/env python3
"""Walk guide: path segmentation from the Raspberry Pi 5 CSI camera.

Captures frames from the CSI camera (via Picamera2 / libcamera), runs YOLO
segmentation to find the path, computes the heading toward it, and announces
the direction by playing sound/left.mp3, sound/right.mp3 or sound/forward.mp3
when the direction changes.

Two display modes (``--mode``):
  * ``screen``   - show an OpenCV window with the mask overlay + heading line
                   (needs a GUI build of OpenCV and a display, e.g. SSH -X).
  * ``terminal`` - no window at all; direction is only logged and spoken.

Inference takes ~0.5 s/frame, so the loop only processes one frame every
PROCESS_INTERVAL seconds rather than running flat out.
"""

import argparse
import logging
import os
import shutil
import subprocess
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
SOUND_DIR = "sound"
SOUND_FILES = {
    "left": os.path.join(SOUND_DIR, "left.mp3"),
    "right": os.path.join(SOUND_DIR, "right.mp3"),
    "forward": os.path.join(SOUND_DIR, "forward.mp3"),
}
# CLI mp3 players tried in order; the first one found on PATH is used.
AUDIO_PLAYERS = [
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpv", "--no-video", "--really-quiet"],
    ["cvlc", "--play-and-exit", "--quiet"],
]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("walkguide")

log.info("Loading model: %s", MODEL_PATH)
model = YOLO(MODEL_PATH, verbose=False)
log.info("Model loaded. Classes: %s", getattr(model, "names", {}))


def find_audio_player():
    """Return the first available CLI mp3 player command, or None."""
    for player in AUDIO_PLAYERS:
        if shutil.which(player[0]):
            return player
    return None


def play_sound(player, command, proc_state):
    """Play the mp3 for `command` ("left"/"right"/"forward"), non-blocking.

    Stops any sound still playing from the previous command so announcements
    don't pile up.
    """
    if player is None:
        return
    path = SOUND_FILES.get(command)
    if not path or not os.path.exists(path):
        log.warning("Sound file missing: %s", path)
        return

    prev = proc_state.get("proc")
    if prev is not None and prev.poll() is None:
        prev.terminate()
    proc_state["proc"] = subprocess.Popen(
        player + [path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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


def compute_heading(frame, model, draw):
    """Run segmentation on a frame.

    Returns (heading_degrees, annotated_frame_or_None). The annotated frame is
    only built when `draw` is True (screen mode); otherwise None is returned to
    skip the drawing work in terminal mode.
    """
    h, w = frame.shape[:2]
    result = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)[0]
    model_names = getattr(model, "names", {})
    vis = frame.copy() if draw else None
    overlay = vis.copy() if draw else None
    midpoints = []

    if result.masks is None or len(result.masks.data) == 0:
        return 90.0, vis

    for mask_index in get_allowed_mask_indices(result, model_names):
        if mask_index >= len(result.masks.data):
            continue

        mask = result.masks.data[mask_index].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if draw:
            overlay[mask > 0] = (0, 255, 0)  # green path overlay

        for row_ratio in SCAN_HEIGHTS:
            y = int(h * row_ratio)
            if y >= h:
                continue
            filled_x = np.where(mask[y, :] > 0)[0]
            if len(filled_x) > 0:
                midpoints.append((int(np.mean(filled_x)), y))

    if draw:
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

    if not midpoints:
        return 90.0, vis

    avg_x = int(np.mean([point[0] for point in midpoints]))
    target_y = min(point[1] for point in midpoints)

    if draw:
        for mx, my in midpoints:
            cv2.circle(vis, (mx, my), 3, (0, 0, 255), -1)
        cv2.line(vis, (w // 2, h), (avg_x, target_y), (255, 0, 0), 2)

    heading = compute_heading_to_point(frame, avg_x, target_y)
    return heading, vis


def command_for_heading(heading):
    """Map a heading to "left", "right" or "forward".

    Heading is 90 deg straight ahead. Below 90 the path is to the right; above
    90 it is to the left; within the deadband it is straight ("forward").
    """
    error = heading - TARGET_HEADING
    if abs(error) <= HEADING_DEADBAND:
        return "forward"
    return "right" if heading < TARGET_HEADING else "left"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("terminal", "screen"),
        default="screen",
        help="terminal: no window (headless); screen: show OpenCV window "
             "(default: screen).",
    )
    args = parser.parse_args()
    show = args.mode == "screen"

    player = find_audio_player()
    if player is None:
        log.warning(
            "No mp3 player found (tried %s). Sounds disabled; install one, "
            "e.g. sudo apt install -y mpg123.",
            ", ".join(p[0] for p in AUDIO_PLAYERS),
        )
    else:
        log.info("Using audio player: %s", player[0])

    log.info("Initialising CSI camera (Picamera2) at %sx%s", *FRAME_SIZE)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    log.info("Camera started in %s mode. Processing 1 frame every %.2fs "
             "(%s Ctrl+C to stop).",
             args.mode, PROCESS_INTERVAL,
             "press 'q' in the window or" if show else "")

    frame_count = 0
    last_command = None
    proc_state = {"proc": None}
    try:
        while True:
            loop_start = time.monotonic()

            # Picamera2 with "RGB888" delivers BGR-ordered data for OpenCV.
            frame = picam2.capture_array()
            frame_count += 1
            if frame_count == 1:
                log.info("First frame read: shape=%s dtype=%s", frame.shape, frame.dtype)

            infer_start = time.monotonic()
            heading, vis = compute_heading(frame, model, draw=show)
            infer_ms = (time.monotonic() - infer_start) * 1000.0
            command = command_for_heading(heading)

            log.debug(
                "frame=%d infer=%.0fms heading=%.1f deg command=%s",
                frame_count, infer_ms, heading, command,
            )

            if command != last_command:
                print(command, flush=True)
                play_sound(player, command, proc_state)
            last_command = command

            if show:
                label = f"heading={heading:.1f}  {command}  {infer_ms:.0f}ms"
                cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, vis)

            # Pace the loop to roughly one frame per PROCESS_INTERVAL.
            elapsed_ms = (time.monotonic() - loop_start) * 1000.0
            wait_ms = max(1, int(PROCESS_INTERVAL * 1000.0 - elapsed_ms))
            if show:
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(wait_ms / 1000.0)
    except KeyboardInterrupt:
        log.info("Interrupted. Read %d frames total.", frame_count)
    finally:
        picam2.stop()
        if show:
            cv2.destroyAllWindows()
        prev = proc_state.get("proc")
        if prev is not None and prev.poll() is None:
            prev.terminate()
        log.info("Camera stopped.")


if __name__ == "__main__":
    main()
