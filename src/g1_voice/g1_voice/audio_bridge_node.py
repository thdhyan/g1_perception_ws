#!/usr/bin/env python3
"""
ROS 2 face of the G1 voice service.

Talks to audio_backend.py over its Unix socket (see that file for why the SDK
lives in a separate process) and maps it onto plain std_msgs topics:

  published:
    /g1/asr/text        std_msgs/String        recognised utterance text
    /g1/asr/event       std_msgs/String        full rt/audio_msg JSON
                                               (text, angle, speaker_id,
                                               sense, confidence, language)
    /g1/audio/play_state std_msgs/Bool         true while the robot is playing

  subscribed:
    /g1/tts/text        std_msgs/String        speak this via onboard TTS
    /g1/audio/play_pcm  std_msgs/UInt8MultiArray  16 kHz mono s16le to speaker
    /g1/audio/volume    std_msgs/Int32         0-100
    /g1/audio/led       std_msgs/ColorRGBA     r/g/b in 0-255, a ignored
    /g1/audio/stop      std_msgs/Empty         stop current playback

Commands are queued onto one worker thread because play_pcm holds the backend
for the length of the audio.
"""
import base64
import json
import queue
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, ColorRGBA, Empty, Int32, String, UInt8MultiArray

DEFAULT_SOCKET_PATH = "/tmp/g1_audio_backend.sock"


class AudioBridgeNode(Node):
    def __init__(self):
        super().__init__("g1_audio_bridge")

        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)
        self.declare_parameter("speaker_id", 1)  # 0 = Chinese voice, 1 = English
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("reconnect_period", 2.0)

        self.socket_path = self.get_parameter("socket_path").value
        self.speaker_id = self.get_parameter("speaker_id").value
        self.min_confidence = self.get_parameter("min_confidence").value
        self.reconnect_period = self.get_parameter("reconnect_period").value

        self.asr_text_pub = self.create_publisher(String, "/g1/asr/text", 10)
        self.asr_event_pub = self.create_publisher(String, "/g1/asr/event", 10)
        self.play_state_pub = self.create_publisher(Bool, "/g1/audio/play_state", 10)

        self.create_subscription(String, "/g1/tts/text", self.on_tts, 10)
        self.create_subscription(UInt8MultiArray, "/g1/audio/play_pcm", self.on_play_pcm, 10)
        self.create_subscription(Int32, "/g1/audio/volume", self.on_volume, 10)
        self.create_subscription(ColorRGBA, "/g1/audio/led", self.on_led, 10)
        self.create_subscription(Empty, "/g1/audio/stop", self.on_stop, 10)

        self._cmd_queue = queue.Queue()
        self._cmd_sock = None
        self._cmd_file = None

        threading.Thread(target=self._command_worker, daemon=True).start()
        threading.Thread(target=self._event_worker, daemon=True).start()

        self.get_logger().info(f"g1_audio_bridge up, backend socket {self.socket_path}")

    # -- backend plumbing ----------------------------------------------------

    def _connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        return sock

    def _command_worker(self):
        while rclpy.ok():
            req = self._cmd_queue.get()
            try:
                if self._cmd_file is None:
                    self._cmd_sock = self._connect()
                    self._cmd_file = self._cmd_sock.makefile("rwb")
                self._cmd_file.write((json.dumps(req) + "\n").encode())
                self._cmd_file.flush()
                line = self._cmd_file.readline()
                if not line:
                    raise ConnectionError("backend closed the connection")
                resp = json.loads(line)
                if not resp.get("ok"):
                    self.get_logger().warn(f"{req['cmd']} failed: {resp}")
            except Exception as e:
                self.get_logger().warn(f"backend command {req.get('cmd')} failed: {e}")
                self._cmd_file = None
                if self._cmd_sock is not None:
                    self._cmd_sock.close()
                    self._cmd_sock = None
                time.sleep(self.reconnect_period)

    def _event_worker(self):
        while rclpy.ok():
            try:
                sock = self._connect()
                rfile = sock.makefile("rwb")
                rfile.write(b'{"cmd": "subscribe"}\n')
                rfile.flush()
                for line in rfile:
                    line = line.strip()
                    if not line:
                        continue
                    self._handle_event(json.loads(line))
            except Exception as e:
                self.get_logger().warn(f"event stream lost ({e}), retrying")
            time.sleep(self.reconnect_period)

    def _handle_event(self, payload: dict):
        event = payload.get("event")
        if event == "asr":
            self.asr_event_pub.publish(String(data=json.dumps(payload)))
            confidence = payload.get("confidence", 1.0)
            if confidence < self.min_confidence:
                return
            text = (payload.get("text") or "").strip()
            if text:
                self.asr_text_pub.publish(String(data=text))
        elif event == "play_state":
            self.play_state_pub.publish(Bool(data=bool(payload.get("play_state", 0))))

    # -- subscriptions -------------------------------------------------------

    def on_tts(self, msg: String):
        text = msg.data.strip()
        if text:
            self._cmd_queue.put({"cmd": "tts", "text": text, "speaker_id": self.speaker_id})

    def on_play_pcm(self, msg: UInt8MultiArray):
        pcm = bytes(bytearray(msg.data))
        if pcm:
            self._cmd_queue.put({
                "cmd": "play_pcm",
                "pcm_b64": base64.b64encode(pcm).decode(),
                "stream_id": str(int(time.time() * 1000)),
            })

    def on_volume(self, msg: Int32):
        self._cmd_queue.put({"cmd": "set_volume", "volume": int(msg.data)})

    def on_led(self, msg: ColorRGBA):
        self._cmd_queue.put({
            "cmd": "led",
            "r": int(msg.r), "g": int(msg.g), "b": int(msg.b),
        })

    def on_stop(self, _msg: Empty):
        """Drop anything still queued as well, otherwise a stop during a long
        reply is followed by the rest of that reply."""
        dropped = 0
        while True:
            try:
                self._cmd_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            self.get_logger().info(f"stop: dropped {dropped} queued commands")
        self._cmd_queue.put({"cmd": "play_stop"})


def main():
    rclpy.init()
    node = AudioBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
