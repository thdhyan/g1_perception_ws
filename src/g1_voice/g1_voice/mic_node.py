#!/usr/bin/env python3
"""
Raw microphone stream from the G1's four-mic array.

The voice service does not expose the microphone through any SDK call. It
multicasts the already-beamformed mono stream on UDP 239.168.123.161:5555 as
raw 16 kHz, mono, 16-bit little-endian PCM (~160 ms frames), which is exactly
what PlayStream and any external ASR want.

  published:
    /g1/mic/audio_raw   std_msgs/UInt8MultiArray  every received frame
    /g1/mic/speech      std_msgs/UInt8MultiArray  one RMS-gated utterance
    /g1/mic/voiced      std_msgs/Bool             voice-activity edge

Only needed when running your own ASR. The robot's built-in offline ASR is
already published by audio_bridge_node on /g1/asr/text.

Joining the multicast group requires an interface on the robot's 192.168.123.x
subnet; set ~local_ip explicitly if the host has several.
"""
import socket
import struct
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, UInt8MultiArray

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


class MicNode(Node):
    def __init__(self):
        super().__init__("g1_mic")

        self.declare_parameter("group_ip", "239.168.123.161")
        self.declare_parameter("port", 5555)
        self.declare_parameter("local_ip", "")          # "" = autodetect 192.168.123.x
        self.declare_parameter("subnet_prefix", "192.168.123.")
        self.declare_parameter("publish_raw", True)
        self.declare_parameter("rms_threshold", 500.0)  # int16 units
        self.declare_parameter("silence_ms", 700)
        self.declare_parameter("min_utterance_ms", 300)
        self.declare_parameter("max_utterance_ms", 15000)

        self.group_ip = self.get_parameter("group_ip").value
        self.port = self.get_parameter("port").value
        self.publish_raw = self.get_parameter("publish_raw").value
        self.rms_threshold = self.get_parameter("rms_threshold").value
        self.silence_ms = self.get_parameter("silence_ms").value
        self.min_utterance_ms = self.get_parameter("min_utterance_ms").value
        self.max_utterance_ms = self.get_parameter("max_utterance_ms").value

        self.raw_pub = self.create_publisher(UInt8MultiArray, "/g1/mic/audio_raw", 10)
        self.speech_pub = self.create_publisher(UInt8MultiArray, "/g1/mic/speech", 10)
        self.voiced_pub = self.create_publisher(Bool, "/g1/mic/voiced", 10)

        self._utterance = bytearray()
        self._silence_ms_acc = 0
        self._voiced = False

        self.sock = self._open_socket()
        threading.Thread(target=self._rx_loop, daemon=True).start()

    def _local_ip(self) -> str:
        configured = self.get_parameter("local_ip").value
        if configured:
            return configured
        prefix = self.get_parameter("subnet_prefix").value
        for iface, addrs in _interface_addresses():
            for addr in addrs:
                if addr.startswith(prefix):
                    self.get_logger().info(f"multicast via {iface} ({addr})")
                    return addr
        raise RuntimeError(
            f"no interface on {prefix}x found; set the local_ip parameter"
        )

    def _open_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(self.group_ip),
            socket.inet_aton(self._local_ip()),
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.get_logger().info(f"listening on {self.group_ip}:{self.port}")
        return sock

    def _rx_loop(self):
        while rclpy.ok():
            data, _ = self.sock.recvfrom(8192)
            if not data:
                continue
            if self.publish_raw:
                self.raw_pub.publish(UInt8MultiArray(data=list(data)))
            self._segment(data)

    def _segment(self, frame: bytes):
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples * samples)))
        frame_ms = (len(frame) / BYTES_PER_SAMPLE) * 1000.0 / SAMPLE_RATE

        if rms >= self.rms_threshold:
            if not self._voiced:
                self._voiced = True
                self.voiced_pub.publish(Bool(data=True))
            self._silence_ms_acc = 0
            self._utterance.extend(frame)
        elif self._utterance:
            self._silence_ms_acc += frame_ms
            self._utterance.extend(frame)

        utterance_ms = (len(self._utterance) / BYTES_PER_SAMPLE) * 1000.0 / SAMPLE_RATE
        ended = self._silence_ms_acc >= self.silence_ms
        overlong = utterance_ms >= self.max_utterance_ms
        if self._utterance and (ended or overlong):
            if utterance_ms >= self.min_utterance_ms:
                self.speech_pub.publish(UInt8MultiArray(data=list(self._utterance)))
            self._utterance = bytearray()
            self._silence_ms_acc = 0
            self._voiced = False
            self.voiced_pub.publish(Bool(data=False))


def _interface_addresses():
    """(iface, [ipv4...]) pairs, without pulling in netifaces."""
    import subprocess

    out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True)
    result = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[2] == "inet":
            result.setdefault(fields[1], []).append(fields[3].split("/")[0])
    return result.items()


def main():
    rclpy.init()
    node = MicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
