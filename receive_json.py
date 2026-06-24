import asyncio
import websockets
import json
import os
import ssl
import argparse


DEFAULT_SERVER = "wss://signaling.ehb.be"
DEFAULT_ROOM = "/ws/pathnavigation"
# Load the bearer token from the environment (set it on the RPi5, e.g. in
# ~/.bashrc or the systemd service: export PATHFINDER_BEARER_TOKEN="...").
DEFAULT_TOKEN = os.environ.get("PATHFINDER_BEARER_TOKEN")


async def receive_messages(server, room, token, use_tls):

    uri = server.rstrip("/") + room

    print("🔌 Connecting to signaling server...")
    print("Server :", server)
    print("Room   :", room)

    #ssl_context = None
    ssl_context = ssl.create_default_context()

    async with websockets.connect(
        uri,
        ssl=ssl_context,
        origin="http://localhost",
        compression=None,
        additional_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Authorization": f"Bearer {token}"
        },
    ) as websocket:

        print(f"✅ Connected to signaling server ({uri})")

        while True:
            try:
                message = await websocket.recv()
                print("📩 Raw message received:", message)

                data = json.loads(message)

                print("📦 Parsed JSON:")
                print("   Type :", data.get("type"))
                print("   From :", data.get("from"))
                print("   Data :", data.get("data"))
                print("-" * 40)

            except websockets.exceptions.ConnectionClosed:
                print("⚠ Connection to server closed.")
                break
            except json.JSONDecodeError:
                print("❌ Could not parse JSON message.")


def main():

    parser = argparse.ArgumentParser(description="WebSocket signaling receiver")

    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Signaling server (default: {DEFAULT_SERVER})"
    )

    parser.add_argument(
        "--room",
        default=DEFAULT_ROOM,
        help=f"Room path (default: {DEFAULT_ROOM})"
    )

    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN,
        help="Bearer token (default: PATHFINDER_BEARER_TOKEN env var)"
    )

    parser.add_argument(
        "--tls",
        action="store_true",
        help="Enable TLS (for wss://)"
    )

    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing bearer token. Set the PATHFINDER_BEARER_TOKEN environment "
            'variable (export PATHFINDER_BEARER_TOKEN="your-token") or pass --token.'
        )

    asyncio.run(receive_messages(args.server, args.room, args.token, args.tls))


if __name__ == "__main__":
    main()