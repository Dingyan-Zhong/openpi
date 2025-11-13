import asyncio
import base64
import http
import io
import json
import logging
import time
import traceback
from PIL import Image

import numpy as np
from openpi_client import base_policy as _base_policy
import websockets.asyncio.server as _server
import websockets.frames
import websockets

logger = logging.getLogger(__name__)


class WebsocketPolicyServerJson:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = json.loads(await websocket.recv())
                for key in obs:
                    if key.endswith("rgb") and isinstance(obs[key], str):
                        obs[key] = _decode_image_to_array(obs[key])
                        if obs[key].shape[0] == 3:
                            obs[key] = np.transpose(obs[key], (1, 2, 0))  # C,H,W to H,W,C

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                print("WebsocketPolicyServerJson action:", action["actions"])
                payload = {
                    "status": "success",
                    "action": action["actions"].tolist(),
                    "server_timing": action.get("server_timing", {}),
                }
                await websocket.send(json.dumps(payload))
                prev_total_time = time.monotonic() - start_time

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                tb = traceback.format_exc()
               # send structured error JSON so client can parse it
                try:
                    await websocket.send(json.dumps({"status": "error", "error": tb}))
                except Exception:
                    # if sending fails, ignore and proceed to close
                    pass
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

# helper: decode a base64 string to a PIL image
def _decode_image(b64_str: str) -> Image.Image:
    # support data URLs too: "data:image/png;base64,..."
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw))

# helper: decode to numpy array (uint8 RGB by default)
def _decode_image_to_array(b64_str: str, mode: str | None = "RGB") -> np.ndarray:
    img = _decode_image(b64_str)
    if mode is not None:
        img = img.convert(mode)
    return np.array(img)
