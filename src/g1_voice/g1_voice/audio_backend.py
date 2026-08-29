#!/usr/bin/env python3
"""
G1 audio/VUI backend -- standalone process, no rclpy.

Owns the single unitree_sdk2py AudioClient connection plus the rt/audio_msg
DDS subscriber, and exposes both over a local Unix domain socket as a
JSON-line service. Runs OUTSIDE any ROS process for the same reason
robot_bridge.py does: rclpy's Node() and unitree_sdk2py's
ChannelFactoryInitialize() each load their own copy of libddsc.so, and
creating both DDS participants in one process segfaults.

Run this once, standalone (not via ros2 run):

    python3 audio_backend.py enp2s0 [socket_path]

Two connection modes, chosen per connection by the first line sent:

  1. command mode (default) -- one JSON request per line, one reply per line:
        {"cmd": "tts", "text": "hello", "speaker_id": 1}   -> {"ok": true}
        {"cmd": "get_volume"}                              -> {"ok": true, "volume": 80}
        {"cmd": "set_volume", "volume": 80}                -> {"ok": true}
        {"cmd": "led", "r": 0, "g": 255, "b": 0}           -> {"ok": true}
        {"cmd": "play_pcm", "pcm_b64": "...", "stream_id": "x"} -> {"ok": true}
        {"cmd": "play_stop"}                               -> {"ok": true}
     Errors: {"ok": false, "error": "..."}

  2. event mode -- send {"cmd": "subscribe"} and the connection then streams
     one JSON event per line, forwarded from the robot's rt/audio_msg topic:
        {"event": "asr", "text": "hello", "angle": 90, "confidence": 0.95, ...}
        {"event": "play_state", "play_state": 1}

Robot-side requirements (Unitree G1 VUI doc):
  - Vui_Service >= 2.0.3.8, Vui Module >= 2.0.0.3.
  - Microphone only publishes ASR when the robot is put in wake-up mode from
    the app or the remote.
  - PlayStream PCM must be 16 kHz, mono, 16-bit signed little-endian.
  - LedControl calls must be spaced by more than 200 ms.
"""
import base64
import json
import os
import socketserver
import sys
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

ASR_TOPIC = "rt/audio_msg"
APP_NAME = "g1_voice"

# PlayStream: 3 seconds of 16 kHz mono s16le, the chunk size Unitree's own
# example uses. Larger chunks get rejected by the voice service. Chunks are
# fed slightly faster than real time so the service always has audio queued
# without building an unbounded cache on minutes-long replies.
CHUNK_BYTES = 96000
CHUNK_SECONDS = 3.0
CHUNK_PACING = 2.5

LED_MIN_INTERVAL = 0.25  # doc says >200 ms between LedControl calls

DEFAULT_SOCKET_PATH = "/tmp/g1_audio_backend.sock"


class G1Audio:
    """AudioClient wrapper + rt/audio_msg fan-out to subscribed sockets."""

    def __init__(self, network_interface: str, domain_id: int = 0, timeout: float = 10.0):
        ChannelFactoryInitialize(domain_id, network_interface)
        self.client = AudioClient()
        self.client.SetTimeout(timeout)
        self.client.Init()

        self._lock = threading.Lock()
        self._last_led = 0.0

        self._subscribers = []          # list of file objects in event mode
        self._subscribers_lock = threading.Lock()

        self._asr_sub = ChannelSubscriber(ASR_TOPIC, String_)
        self._asr_sub.Init(self._on_audio_msg, 10)

    # -- rt/audio_msg fan-out ------------------------------------------------

    def _on_audio_msg(self, msg: String_):
        """rt/audio_msg carries both ASR results and playback state, both as
        JSON strings; they are told apart by their keys."""
        try:
            payload = json.loads(msg.data)
        except (ValueError, AttributeError):
            print(f"[G1Audio] unparsable rt/audio_msg: {msg}")
            return

        if "play_state" in payload:
            payload["event"] = "play_state"
        elif "text" in payload:
            payload["event"] = "asr"
        else:
            payload["event"] = "unknown"

        self.broadcast(payload)

    def broadcast(self, payload: dict):
        line = (json.dumps(payload) + "\n").encode()
        with self._subscribers_lock:
            dead = []
            for wfile in self._subscribers:
                try:
                    wfile.write(line)
                    wfile.flush()
                except (BrokenPipeError, ValueError, OSError):
                    dead.append(wfile)
            for wfile in dead:
                self._subscribers.remove(wfile)

    def add_subscriber(self, wfile):
        with self._subscribers_lock:
            self._subscribers.append(wfile)

    def remove_subscriber(self, wfile):
        with self._subscribers_lock:
            if wfile in self._subscribers:
                self._subscribers.remove(wfile)

    # -- AudioClient API -----------------------------------------------------

    def tts(self, text: str, speaker_id: int = 1) -> int:
        """speaker_id 0 = Chinese voice, 1 = English voice. The service does
        not support mixed Chinese/English text in one call."""
        return self.client.TtsMaker(text, speaker_id)

    def get_volume(self) -> int:
        code, data = self.client.GetVolume()
        if code != 0:
            raise RuntimeError(f"GetVolume failed, code={code}")
        return int(data["volume"])

    def set_volume(self, volume: int) -> int:
        return self.client.SetVolume(max(0, min(100, int(volume))))

    def led(self, r: int, g: int, b: int) -> int:
        wait = LED_MIN_INTERVAL - (time.time() - self._last_led)
        if wait > 0:
            time.sleep(wait)
        self._last_led = time.time()
        return self.client.LedControl(int(r), int(g), int(b))

    def play_pcm(self, pcm: bytes, stream_id: str) -> int:
        """Same stream_id keeps appending to the playback cache, a new one
        interrupts whatever is playing."""
        ret = 0
        for offset in range(0, len(pcm), CHUNK_BYTES):
            ret = self.client.PlayStream(APP_NAME, stream_id, pcm[offset:offset + CHUNK_BYTES])
            if ret != 0:
                break
            time.sleep(CHUNK_PACING)
        return ret

    def play_stop(self) -> int:
        return self.client.PlayStop(APP_NAME)


def dispatch(audio: G1Audio, req: dict) -> dict:
    cmd = req.get("cmd")
    with audio._lock:
        if cmd == "tts":
            ret = audio.tts(str(req.get("text", "")), int(req.get("speaker_id", 1)))
            return {"ok": ret == 0, "ret": ret}
        elif cmd == "get_volume":
            return {"ok": True, "volume": audio.get_volume()}
        elif cmd == "set_volume":
            ret = audio.set_volume(req.get("volume", 80))
            return {"ok": ret == 0, "ret": ret}
        elif cmd == "led":
            ret = audio.led(req.get("r", 0), req.get("g", 0), req.get("b", 0))
            return {"ok": ret == 0, "ret": ret}
        elif cmd == "play_pcm":
            pcm = base64.b64decode(req["pcm_b64"])
            stream_id = str(req.get("stream_id") or int(time.time() * 1000))
            ret = audio.play_pcm(pcm, stream_id)
            return {"ok": ret == 0, "ret": ret, "bytes": len(pcm)}
        elif cmd == "play_stop":
            return {"ok": True, "ret": audio.play_stop()}
        elif cmd == "ping":
            return {"ok": True}
        else:
            return {"ok": False, "error": f"unknown cmd: {cmd}"}


def make_handler(audio: G1Audio):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            subscribed = False
            for line in self.rfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except ValueError as e:
                    self.wfile.write((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
                    continue

                if req.get("cmd") == "subscribe":
                    if not subscribed:
                        audio.add_subscriber(self.wfile)
                        subscribed = True
                    self.wfile.write((json.dumps({"ok": True, "subscribed": True}) + "\n").encode())
                    continue

                try:
                    resp = dispatch(audio, req)
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}
                self.wfile.write((json.dumps(resp) + "\n").encode())

            if subscribed:
                audio.remove_subscriber(self.wfile)

    return Handler


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface [socket_path]")
        sys.exit(-1)

    network_interface = sys.argv[1]
    socket_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SOCKET_PATH

    print(f"Connecting to G1 voice service on {network_interface}...")
    audio = G1Audio(network_interface)
    print(f"Connected. Volume: {audio.get_volume()}")

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = ThreadedUnixServer(socket_path, make_handler(audio))
    print(f"Listening on {socket_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if os.path.exists(socket_path):
            os.remove(socket_path)


if __name__ == "__main__":
    main()
