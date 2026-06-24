import asyncio
import json
import os
import ssl
import websockets
import secrets
import time

ROOM = "/ws/pathnavigation"
#SIGNALING_SERVER = f"ws://192.168.0.81:9000{ROOM}"
SIGNALING_SERVER = f"wss://signaling.ehb.be{ROOM}"
SESSION_ID = "demo-session-001"

#BEARER_TOKEN = secrets.token_urlsafe(32) ; print(BEARER_TOKEN) # Generate a new bearer token

# Load the bearer token from the environment (set it on the RPi5, e.g. in
# ~/.bashrc or the systemd service: export PATHFINDER_BEARER_TOKEN="...").
BEARER_TOKEN = os.environ.get("PATHFINDER_BEARER_TOKEN")
if not BEARER_TOKEN:
    raise SystemExit(
        "Missing PATHFINDER_BEARER_TOKEN environment variable. "
        'Set it with: export PATHFINDER_BEARER_TOKEN="your-token"'
    )

async def send_message():

    message = {
        "sessionId": SESSION_ID,
        "type": "topic",
        "from": "client1",
        "to": "receiver1", # or "all"
        "data": {
            "name": "topic1",
            "value": time.time()
        }
    }

    #ssl_context = None
    ssl_context = ssl.create_default_context() # Uncomment if using wss://


    uri = SIGNALING_SERVER  # Zelfde server als sender
    async with websockets.connect(uri,
        ssl=ssl_context,   # Uncomment if using wss://
        origin="http://localhost",
        compression=None,
        additional_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Authorization": f"Bearer {BEARER_TOKEN}"
        },
    ) as websocket:

        print("Connected")

        await websocket.send(json.dumps(message))
        print("Message sent")

        # wacht op antwoorden
        #async for msg in websocket:
        #    print("Received:", msg)


asyncio.run(send_message())