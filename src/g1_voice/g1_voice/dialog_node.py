#!/usr/bin/env python3
"""
LLM dialog turn: ASR text in, TTS text out.

  subscribed:
    /g1/asr/text          std_msgs/String   utterance from the onboard ASR
    /g1/audio/play_state  std_msgs/Bool     used to drop the robot's own voice

  published:
    /g1/tts/text          std_msgs/String   reply for audio_bridge_node to speak
    /g1/audio/led         std_msgs/ColorRGBA  listening / thinking / speaking cue

Backends are tried in the order given by the llm_chain parameter, a JSON list
of {"name", "kind", "host", "model"} entries where kind is "ollama" or
"openai". The first one that answers wins; one that fails (unreachable, HTTP
error, timeout) is skipped for retry_after seconds before being tried again.
The default chain is robot-onboard Ollama, then the lab server, then OpenAI,
so the robot keeps talking when it is untethered and when the lab is down.

Note that onboard is first for *availability*, not speed: the round trip to
the lab server is about a millisecond, so a small model on the lab GPU will
almost always answer faster than a bigger one on the robot's Jetson.

Requests run on a worker thread; while one is in flight new utterances are
dropped rather than queued, so the robot never replies to a stale question.

The onboard TTS has no mixed-language voice, so keep replies in one language.
Long replies play as one uninterruptible blob; publish std_msgs/Empty on
/g1/audio/stop to cut one short.
"""
import json
import os
import threading
import time

import rclpy
import requests
from rclpy.node import Node
from std_msgs.msg import Bool, ColorRGBA, String

DEFAULT_SYSTEM_PROMPT = (
    "You are a Unitree G1 robot at the Reality Capture Lab at the University "
    "of Minnesota, and you are being developed at the College of Science and "
    "Engineering. Your replies are spoken aloud through the robot's speaker, "
    "so write plain spoken English: no markdown, no emoji, no lists, no code. "
    "Answer in two or three sentences unless you are asked to explain "
    "something in depth."
)

DEFAULT_CHAIN = [
    {"name": "onboard", "kind": "ollama",
     "host": "http://192.168.123.164:11434", "model": "gemma4:e2b"},
    {"name": "lab", "kind": "ollama",
     "host": "http://128.101.125.152:11434", "model": "gemma4:latest"},
    {"name": "openai", "kind": "openai",
     "host": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
]

LED_IDLE = (0, 0, 30)
LED_THINKING = (60, 40, 0)
LED_SPEAKING = (0, 60, 0)


class DialogNode(Node):
    def __init__(self):
        super().__init__("g1_dialog")

        self.declare_parameter("llm_chain", json.dumps(DEFAULT_CHAIN))
        self.declare_parameter("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.declare_parameter("history_turns", 6)
        self.declare_parameter("max_reply_chars", 2000)
        self.declare_parameter("connect_timeout", 2.0)
        self.declare_parameter("request_timeout", 60.0)
        self.declare_parameter("retry_after", 60.0)
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("wake_word", "")       # "" = respond to everything
        self.declare_parameter("min_chars", 2)
        self.declare_parameter("use_led", True)

        self.chain = json.loads(self.get_parameter("llm_chain").value)
        self.history_turns = self.get_parameter("history_turns").value
        self.max_reply_chars = self.get_parameter("max_reply_chars").value
        self.connect_timeout = self.get_parameter("connect_timeout").value
        self.timeout = self.get_parameter("request_timeout").value
        self.retry_after = self.get_parameter("retry_after").value
        self.wake_word = self.get_parameter("wake_word").value.lower().strip()
        self.min_chars = self.get_parameter("min_chars").value
        self.use_led = self.get_parameter("use_led").value

        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self._down_until = {}   # chain entry name -> monotonic time it may be retried

        self.tts_pub = self.create_publisher(String, "/g1/tts/text", 10)
        self.led_pub = self.create_publisher(ColorRGBA, "/g1/audio/led", 10)
        self.create_subscription(String, "/g1/asr/text", self.on_asr, 10)
        self.create_subscription(Bool, "/g1/audio/play_state", self.on_play_state, 10)

        self._history = []
        self._busy = threading.Lock()
        self._speaking = False

        self._set_led(LED_IDLE)
        order = " -> ".join(e.get("name", e["kind"]) for e in self.chain)
        self.get_logger().info(f"g1_dialog up, backend chain: {order}")

    # -- callbacks -----------------------------------------------------------

    def on_play_state(self, msg: Bool):
        self._speaking = msg.data
        if self.use_led:
            self._set_led(LED_SPEAKING if msg.data else LED_IDLE)

    def on_asr(self, msg: String):
        text = msg.data.strip()
        if len(text) < self.min_chars:
            return
        if self._speaking:
            self.get_logger().debug(f"ignoring '{text}' while speaking")
            return
        if self.wake_word:
            if self.wake_word not in text.lower():
                return
            text = text.lower().split(self.wake_word, 1)[1].strip() or text
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn(f"dropping '{text}', a reply is still generating")
            return
        threading.Thread(target=self._respond, args=(text,), daemon=True).start()

    def _respond(self, text: str):
        try:
            self._set_led(LED_THINKING)
            self.get_logger().info(f"user: {text}")
            reply = self._complete(text)
            reply = reply.strip()[: self.max_reply_chars]
            if not reply:
                return
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": reply})
            self._history = self._history[-2 * self.history_turns:]
            self.get_logger().info(f"robot: {reply}")
            self.tts_pub.publish(String(data=reply))
        except Exception as e:
            self.get_logger().error(f"LLM request failed: {e}")
            self._set_led(LED_IDLE)
        finally:
            self._busy.release()

    # -- backends ------------------------------------------------------------

    def _messages(self, text: str):
        system_prompt = self.get_parameter("system_prompt").value
        return (
            [{"role": "system", "content": system_prompt}]
            + self._history
            + [{"role": "user", "content": text}]
        )

    def _complete(self, text: str) -> str:
        messages = self._messages(text)
        errors = []
        for entry in self.chain:
            name = entry.get("name", entry.get("kind"))
            if self._down_until.get(name, 0.0) > time.monotonic():
                continue
            if entry["kind"] == "openai" and not self.api_key:
                errors.append(f"{name}: OPENAI_API_KEY is unset")
                continue
            try:
                started = time.monotonic()
                reply = self._call(entry, messages)
                self.get_logger().info(
                    f"{name} answered in {time.monotonic() - started:.1f}s"
                )
                return reply
            except Exception as e:
                self._down_until[name] = time.monotonic() + self.retry_after
                self.get_logger().warn(f"{name} unavailable ({e}), trying next")
                errors.append(f"{name}: {e}")
        raise RuntimeError("no LLM backend answered -- " + "; ".join(errors))

    def _call(self, entry: dict, messages: list) -> str:
        host = entry["host"].rstrip("/")
        temperature = self.get_parameter("temperature").value
        timeouts = (self.connect_timeout, self.timeout)

        if entry["kind"] == "ollama":
            body = {
                "model": entry["model"],
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            }
            r = requests.post(f"{host}/api/chat", json=body, timeout=timeouts)
            r.raise_for_status()
            return r.json()["message"]["content"]

        if entry["kind"] == "openai":
            body = {
                "model": entry["model"],
                "messages": messages,
                "temperature": temperature,
            }
            r = requests.post(
                f"{host}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeouts,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        raise ValueError(f"unknown backend kind: {entry['kind']}")

    # -- helpers -------------------------------------------------------------

    def _set_led(self, rgb):
        if not self.use_led:
            return
        r, g, b = rgb
        self.led_pub.publish(ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0))


def main():
    rclpy.init()
    node = DialogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
