#!/usr/bin/env python3
"""Walk guide client that offloads inference to the EHB signaling server.

Grabs frames from the Raspberry Pi 5 CSI camera (Picamera2 / libcamera),
streams them to the signaling server where ``segmentVideoServer.py`` runs the
YOLO segmentation, receives the computed heading back, and announces the
direction by playing sound/left.mp3, sound/right.mp3 or sound/forward.mp3 when
it changes.

Protocol (matches segmentVideoServer.py):
  1. send a JSON ``frame_meta`` message (frame_id, sessionId, model, ...)
  2. send the JPEG-encoded frame as raw binary bytes
  3. receive a JSON response containing ``heading`` (and optionally
     ``marker_heading``)

The bearer token is read from the PATHFINDER_BEARER_TOKEN environment variable.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import ssl
import subprocess
import time
from pathlib import Path

import cv2

try:
    from picamera2 import Picamera2
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: picamera2. Install it with: "
        "sudo apt install -y python3-picamera2"
    ) from exc

try:
    import websockets
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: websockets. Install it with: pip install websockets"
    ) from exc


# --- Signaling / streaming configuration -----------------------------------
# Must match the endpoint segmentVideoServer.py connects to so both ends share
# the same signaling room.
SIGNALING_SERVER = "wss://signaling.ehb.be"
SESSION_ID = "rpi5-walkguide-001"
MODEL_NAME = "denham"        # model the server should use (must exist server-side)
DETECTION_CONFIDENCE = 0.5
FRAME_SIZE = (640, 480)
JPEG_QUALITY = 50
SEND_INTERVAL = 0.1         # seconds between frames sent to the server
REPEAT_INTERVAL = 2.0       # re-announce the same direction every N seconds
TARGET_HEADING = 90.0
HEADING_DEADBAND = 2.0

BEARER_TOKEN = os.environ.get("PATHFINDER_BEARER_TOKEN")

# --- Audio -----------------------------------------------------------------
SOUND_DIR = Path(__file__).resolve().parent / "sound"
SOUND_FILES = {
    "left": str(SOUND_DIR / "left.mp3"),
    "right": str(SOUND_DIR / "right.mp3"),
    "forward": str(SOUND_DIR / "forward.mp3"),
    "started": str(SOUND_DIR / "application_started.mp3"),
}
AUDIO_PLAYERS = [
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpv", "--no-video", "--really-quiet"],
    ["cvlc", "--play-and-exit", "--quiet"],
]

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("walkguide-signaling")


def find_audio_player():
    """Return the first available CLI mp3 player command, or None."""
    for player in AUDIO_PLAYERS:
        if shutil.which(player[0]):
            return player
    return None


def play_sound(player, command):
    """Play the mp3 for `command` to completion (blocking)."""
    if player is None:
        return
    path = SOUND_FILES.get(command)
    if not path or not os.path.exists(path):
        log.warning("Sound file missing: %s", path)
        return
    try:
        subprocess.run(
            player + [path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        log.warning("Failed to play %s: %s", path, exc)


def command_for_heading(heading):
    """Map a heading to "left", "right", or None (straight, not announced).

    Heading is 90 deg straight ahead. Below 90 the path is to the right; above
    90 it is to the left. Within the deadband (straight ahead) we return None
    and stay silent - only left and right are announced.
    """
    error = heading - TARGET_HEADING
    if abs(error) <= HEADING_DEADBAND:
        return None
    return "right" if heading < TARGET_HEADING else "left"


async def receive_headings(ws, player, send_times, loop):
    """Consume server responses and announce direction changes.

    The current direction is repeated every REPEAT_INTERVAL seconds so a blind
    user keeps being reminded which way to go, not only when it changes.
    """
    last_command = None
    last_announced_at = 0.0
    async for msg in ws:
        if not isinstance(msg, str):
            continue
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            continue

        # Prefer the marker heading (aruco target) when the server reports one.
        heading = payload.get("marker_heading")
        if heading is None:
            heading = payload.get("heading")
        if heading is None:
            continue

        frame_id = payload.get("frame_id")
        sent_at = send_times.pop(frame_id, None)
        latency_ms = (time.monotonic() - sent_at) * 1000.0 if sent_at else None

        command = command_for_heading(float(heading))
        print(
            f"heading={float(heading):.1f} command={command or 'straight'} "
            f"frame={frame_id} "
            f"latency={f'{latency_ms:.0f}ms' if latency_ms is not None else 'n/a'}",
            flush=True,
        )

        # Only announce left/right; ignore straight (forward) and don't let it
        # reset last_command, so we re-announce only when the turn changes.
        if command is None:
            continue

        now = time.monotonic()
        changed = command != last_command
        due_for_repeat = (now - last_announced_at) >= REPEAT_INTERVAL
        if changed or due_for_repeat:
            print(command, flush=True)
            # Blocking playback off the event loop so streaming continues.
            await loop.run_in_executor(None, play_sound, player, command)
            last_announced_at = time.monotonic()
        last_command = command


async def stream_frames(ws, picam2, send_times, loop):
    """Capture, JPEG-encode and stream frames to the server."""
    frame_id = 0
    last_latency_ms = None
    while True:
        loop_start = time.monotonic()

        # Picamera2 capture + JPEG encode are blocking; keep them off the loop.
        frame = await loop.run_in_executor(None, picam2.capture_array)
        ok, jpeg = await loop.run_in_executor(
            None,
            lambda: cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            ),
        )
        if not ok:
            log.warning("JPEG encode failed; skipping frame.")
            continue

        meta = {
            "type": "frame_meta",
            "frame_id": frame_id,
            "sessionId": SESSION_ID,
            "model": MODEL_NAME,
            "confidence": DETECTION_CONFIDENCE,
            "returnMasks": True,
            "sendMQTT": False,
            "lastlatency": last_latency_ms,
        }
        await ws.send(json.dumps(meta))
        await ws.send(jpeg.tobytes())
        send_times[frame_id] = time.monotonic()
        frame_id += 1

        # Keep send_times from growing unbounded if responses are dropped.
        if len(send_times) > 100:
            for old_id in sorted(send_times)[:-100]:
                send_times.pop(old_id, None)

        elapsed = time.monotonic() - loop_start
        last_latency_ms = round(elapsed * 1000.0, 1)
        await asyncio.sleep(max(0.0, SEND_INTERVAL - elapsed))


async def run():
    player = find_audio_player()
    if player is None:
        log.warning(
            "No mp3 player found (tried %s). Sounds disabled; install one, "
            "e.g. sudo apt install -y mpg123.",
            ", ".join(p[0] for p in AUDIO_PLAYERS),
        )
    else:
        log.info("Using audio player: %s", player[0])

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, play_sound, player, "started")

    log.info("Initialising CSI camera (Picamera2) at %sx%s", *FRAME_SIZE)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    ssl_context = ssl.create_default_context()
    send_times = {}

    log.info("Connecting to signaling server (%s)...", SIGNALING_SERVER)
    try:
        async with websockets.connect(
            SIGNALING_SERVER,
            ssl=ssl_context,
            origin="http://localhost",
            compression=None,
            additional_headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                "Authorization": f"Bearer {BEARER_TOKEN}",
            },
        ) as ws:
            log.info("Connected. Streaming frames (Ctrl+C to stop).")
            await asyncio.gather(
                stream_frames(ws, picam2, send_times, loop),
                receive_headings(ws, player, send_times, loop),
            )
    finally:
        picam2.stop()
        log.info("Camera stopped.")


def main():
    global SIGNALING_SERVER, MODEL_NAME, SESSION_ID

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=SIGNALING_SERVER,
                        help=f"Signaling server (default: {SIGNALING_SERVER})")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Model name for the server (default: {MODEL_NAME})")
    parser.add_argument("--session", default=SESSION_ID,
                        help=f"Session id (default: {SESSION_ID})")
    args = parser.parse_args()

    SIGNALING_SERVER = args.server
    MODEL_NAME = args.model
    SESSION_ID = args.session

    if not BEARER_TOKEN:
        raise SystemExit(
            "Missing PATHFINDER_BEARER_TOKEN environment variable. "
            'Set it with: export PATHFINDER_BEARER_TOKEN="your-token"'
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted.")


if __name__ == "__main__":
    main()
