# g1_voice

Voice interaction for the Unitree G1: onboard ASR in, LLM in the middle,
onboard TTS out, all as ordinary ROS 2 topics.

## Hardware / service facts

From the Unitree VuiClient service docs and `unitree_sdk2`:

- Four-mic array (20 mm spacing), 8 Ω 3 W speaker, 256-colour RGB strip.
- Requires `Vui_Service >= 2.0.3.8` and `Vui Module >= 2.0.0.3`.
- The microphone only produces ASR when the robot is switched into **wake-up
  mode** from the app or the remote.
- ASR is local, offline, non-streaming by default. Results arrive on the DDS
  topic `rt/audio_msg` (`std_msgs::msg::dds_::String_`) as JSON with
  `text`, `angle` (0-180 azimuth), `speaker_id`, `sense` (emotion),
  `confidence`, `language`, `is_final`. The same topic carries
  `{"play_state": 0|1}`.
- TTS is local and offline. `speaker_id` 0 = Chinese voice, 1 = English.
  Mixed Chinese/English text in one call is not supported.
- `PlayStream` PCM must be **16 kHz, mono, 16-bit signed little-endian**. Reusing
  a `stream_id` appends to the playback cache; a new one interrupts playback.
- `LedControl` calls must be spaced by more than 200 ms (enforced in the backend).
- The raw microphone is not in the SDK at all. It is multicast on UDP
  `239.168.123.161:5555` as raw 16 kHz mono s16le.

## Why there are two processes

`rclpy.Node()` and `unitree_sdk2py`'s `ChannelFactoryInitialize()` each load
their own copy of `libddsc.so`, and creating both DDS participants in one
process segfaults — the same constraint that produced
`g1_control/robot_bridge.py`. So the SDK lives in `audio_backend.py`, a plain
Python process with no rclpy, and the ROS nodes reach it over a Unix socket.

```
                      DDS (robot net)              Unix socket           ROS 2
G1 voice service <--------------------> audio_backend.py <--------> audio_bridge_node
rt/audio_msg     -------------------->                                    |
                                                                          v
UDP 239.168.123.161:5555 ------------------------------> mic_node    dialog_node
```

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/g1/asr/text` | `std_msgs/String` | out — recognised utterance |
| `/g1/asr/event` | `std_msgs/String` | out — full ASR JSON incl. `angle` |
| `/g1/audio/play_state` | `std_msgs/Bool` | out — true while speaking |
| `/g1/tts/text` | `std_msgs/String` | in — speak via onboard TTS |
| `/g1/audio/play_pcm` | `std_msgs/UInt8MultiArray` | in — 16 kHz mono s16le |
| `/g1/audio/volume` | `std_msgs/Int32` | in — 0-100 |
| `/g1/audio/led` | `std_msgs/ColorRGBA` | in — r/g/b in 0-255 |
| `/g1/audio/stop` | `std_msgs/Empty` | in — stop playback |
| `/g1/mic/audio_raw` | `std_msgs/UInt8MultiArray` | out — raw mic frames |
| `/g1/mic/speech` | `std_msgs/UInt8MultiArray` | out — one segmented utterance |
| `/g1/mic/voiced` | `std_msgs/Bool` | out — VAD edge |

## Running

Build:

```bash
cd ~/Projects/thesis/g1_perception_ws
colcon build --packages-select g1_voice
source install/setup.bash
```

Start the backend once, standalone, with the robot's network interface:

```bash
python3 src/g1_voice/g1_voice/audio_backend.py enp2s0
```

Then the ROS side:

```bash
ros2 launch g1_voice voice.launch.py            # bridge + dialog
ros2 launch g1_voice voice.launch.py use_mic:=true   # also raw mic stream
```

Smoke tests without the LLM:

```bash
ros2 topic pub --once /g1/tts/text std_msgs/String "{data: 'Hello, I am G1.'}"
ros2 topic pub --once /g1/audio/volume std_msgs/Int32 "{data: 80}"
ros2 topic echo /g1/asr/text
```

## LLM backends

`dialog_node` walks the `llm_chain` parameter in order and uses the first
backend that answers. A backend that fails is skipped for `retry_after`
seconds, so a dead host costs one `connect_timeout` (2 s), not one per turn.

Default chain:

1. `onboard` — Ollama on the robot's Jetson, `http://192.168.123.164:11434`
2. `lab` — the lab server, `http://128.101.125.152:11434`
3. `openai` — `https://api.openai.com/v1`, needs `OPENAI_API_KEY` in the env

Onboard is first for **availability when untethered**, not for speed: the lab
server is about a millisecond away over ethernet, and a small model on the lab
GPU beats a bigger one on the Jetson. Measured on this stack, `gemma4:latest`
on the lab server takes 20-33 s for a three-sentence reply, which is far too
slow for conversation — pick a 3-4B model on either host.

Only `requests` is used, so no vendor SDK is needed.

### Ollama on the robot

The Jetson runs Ubuntu 20.04 / Python 3.8 / CUDA 11.4, so keep the model small.
`gemma4:e2b` (~2B effective, ~3 GB at q4) is the demo default:

```bash
ssh unitree@192.168.123.164
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e2b        # verify the tag exists; fall back to a 3-4B q4 model
sudo systemctl edit ollama    # OLLAMA_HOST=0.0.0.0:11434 so the laptop can reach it
```

Ollama ships its own CUDA runtime, so it does not care that the host is on
JetPack 5 — unlike Isaac ROS or `vllm-omni`, which need CUDA 12+ and therefore
a JetPack 7 reflash. Reflashing the robot's Orin NX wipes Unitree's own
services, `vui` included, so it takes this package down with it.

### Robot compute budget (Orin NX 16 GB, unified CPU/GPU memory)

| Component | Unified RAM |
|---|---|
| Ubuntu + Unitree services + Livox/RealSense drivers | ~3-4 GB |
| cuVSLAM | ~1-1.5 GB |
| nvblox (room at 5 cm, TSDF+ESDF) | ~2-4 GB |
| VoxelNeXt | ~2-3 GB |
| `gemma4:e4b` q4 + KV | ~5 GB |
| Cosmos3-Edge (BF16, the only supported precision) | ~8-9 GB |

Everything at once is ~22-26 GB and does not fit; 100 TOPS is the second wall.
Workable splits: perception plus `gemma4:e2b` on the robot with Cosmos3-Edge on
the lab server, or voice plus a small VLM on the robot with perception left on
the laptop. Cosmos3-Edge on-robot is only realistic on a Jetson Thor payload,
which is the configuration NVIDIA's own G1 tutorial assumes.

## Long replies

`max_reply_chars` defaults to 2000, about two minutes of speech at 150 wpm.
Two consequences:

- The backend feeds `PlayStream` slightly faster than real time (3 s chunks
  every 2.5 s) so minutes-long audio never builds an unbounded service-side
  cache.
- A long reply cannot be interrupted by talking over it. Publishing to
  `/g1/audio/stop` stops playback *and* drops anything still queued in the
  bridge, so it is the one reliable way to cut the robot off.

## Custom TTS (Kokoro, Piper, ...)

`PlayStream` is direct speaker access — raw PCM straight to the 8 Ω speaker.
The speaker hangs off the VUI audio board, not an ALSA device on the Jetson,
so there is no lower-level path and none is needed.

Run the TTS **on the laptop**, not the robot: Kokoro needs Python >= 3.10 and a
current torch, while the robot is on Python 3.8 with JetPack-pinned CUDA 11.4.
The laptop has Python 3.12 and an RTX 4060. Pipeline:

```
/g1/tts/text -> kokoro (24 kHz float32) -> resample 16 kHz -> int16 LE
             -> /g1/audio/play_pcm -> PlayStream -> speaker
```

Not implemented yet; when it is, it replaces `/g1/tts/text` consumption only —
every other topic stays the same.

## Known limits

- Replies are spoken as one blocking TTS call; there is no streaming
  sentence-by-sentence playback yet.
- Barge-in is handled by ignoring ASR while `/g1/audio/play_state` is true, so
  the robot cannot be interrupted mid-sentence except via `/g1/audio/stop`.
- The onboard TTS voice cannot be changed beyond the Chinese/English role; see
  the custom TTS section above. The demo uses the built-in TTS.
- `mic_node` uses a plain RMS gate, not a real VAD.
- Nothing calls the locomotion bridge yet; tool-calling from `dialog_node` into
  `g1_control/robot_bridge.py` is the obvious next step.
