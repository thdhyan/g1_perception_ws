# g1_voice

Voice for Unitree G1: onboard ASR in, LLM middle, onboard TTS out — all ROS 2 topics.

## Hardware / service facts

From Unitree VuiClient service docs + `unitree_sdk2`:

- Four-mic array (20 mm spacing), 8 Ω 3 W speaker, 256-colour RGB strip.
- Need `Vui_Service >= 2.0.3.8` + `Vui Module >= 2.0.0.3`.
- Mic only give ASR when robot switch into **wake-up mode** from app or remote.
- ASR local, offline, non-streaming default. Results land DDS topic
  `rt/audio_msg` (`std_msgs::msg::dds_::String_`) as JSON w/
  `text`, `angle` (0-180 azimuth), `speaker_id`, `sense` (emotion),
  `confidence`, `language`, `is_final`. Same topic carry
  `{"play_state": 0|1}`.
- TTS local, offline. `speaker_id` 0 = Chinese voice, 1 = English.
  Mixed Chinese/English one call — not supported.
- `PlayStream` PCM must be **16 kHz, mono, 16-bit signed little-endian**. Reuse
  `stream_id` appends to playback cache; new one interrupt playback.
- `LedControl` calls need spacing >200 ms (backend-enforced).
- Raw mic not in SDK at all. Multicast on UDP
  `239.168.123.161:5555` as raw 16 kHz mono s16le.

## Why two processes

`rclpy.Node()` + `unitree_sdk2py`'s `ChannelFactoryInitialize()` each load
own copy `libddsc.so` — both DDS participants one process = segfault. Same
constraint that made `g1_control/robot_bridge.py`. So SDK live in
`audio_backend.py`, plain Python process no rclpy, ROS nodes reach it via
Unix socket.

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

Start backend once, standalone, w/ robot's network interface:

```bash
python3 src/g1_voice/g1_voice/audio_backend.py enp2s0
```

Then ROS side:

```bash
ros2 launch g1_voice voice.launch.py            # bridge + dialog
ros2 launch g1_voice voice.launch.py use_mic:=true   # also raw mic stream
```

Smoke tests, no LLM:

```bash
ros2 topic pub --once /g1/tts/text std_msgs/String "{data: 'Hello, I am G1.'}"
ros2 topic pub --once /g1/audio/volume std_msgs/Int32 "{data: 80}"
ros2 topic echo /g1/asr/text
```

## LLM backends

`dialog_node` walks `llm_chain` param in order, uses first backend answers.
Failed backend skipped for `retry_after` seconds — dead host cost one
`connect_timeout` (2 s), not one per turn.

Default chain:

1. `onboard` — Ollama on robot's Jetson, `http://192.168.123.164:11434`
2. `lab` — lab server, `http://128.101.125.152:11434`
3. `openai` — `https://api.openai.com/v1`, needs `OPENAI_API_KEY` in env

Onboard first for **availability when untethered**, not speed: lab server
~1 ms away over ethernet, small model on lab GPU beats bigger one on Jetson.
Measured this stack, `gemma4:latest` on lab server take 20-33 s for
three-sentence reply — too slow for conversation. Pick 3-4B model either host.

Only `requests` used — no vendor SDK needed.

### Ollama on robot

Jetson run Ubuntu 20.04 / Python 3.8 / CUDA 11.4 — keep model small.
`gemma4:e2b` (~2B effective, ~3 GB at q4) demo default:

```bash
ssh unitree@192.168.123.164
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e2b        # verify the tag exists; fall back to a 3-4B q4 model
sudo systemctl edit ollama    # OLLAMA_HOST=0.0.0.0:11434 so the laptop can reach it
```

Ollama ship own CUDA runtime — doesn't care host on JetPack 5, unlike
Isaac ROS or `vllm-omni` which need CUDA 12+ (JetPack 7 reflash). Reflash
robot's Orin NX wipe Unitree's own services, `vui` included — takes package
down too.

### Robot compute budget (Orin NX 16 GB, unified CPU/GPU memory)

| Component | Unified RAM |
|---|---|
| Ubuntu + Unitree services + Livox/RealSense drivers | ~3-4 GB |
| cuVSLAM | ~1-1.5 GB |
| nvblox (room at 5 cm, TSDF+ESDF) | ~2-4 GB |
| VoxelNeXt | ~2-3 GB |
| `gemma4:e4b` q4 + KV | ~5 GB |
| Cosmos3-Edge (BF16, only supported precision) | ~8-9 GB |

Everything at once ~22-26 GB, no fit; 100 TOPS second wall. Workable splits:
perception + `gemma4:e2b` on robot w/ Cosmos3-Edge on lab server, or voice +
small VLM on robot w/ perception left on laptop. Cosmos3-Edge on-robot only
realistic on Jetson Thor payload — config NVIDIA's own G1 tutorial assumes.

## Long replies

`max_reply_chars` default 2000, ~two minutes speech at 150 wpm. Two
consequences:

- Backend feed `PlayStream` slightly faster than real time (3 s chunks
  every 2.5 s) so minutes-long audio never build unbounded service-side
  cache.
- Long reply can't be interrupted by talking over it. Publish
  `/g1/audio/stop` stops playback *and* drops anything still queued in
  bridge — one reliable way to cut robot off.

## Custom TTS (Kokoro, Piper, ...)

`PlayStream` = direct speaker access — raw PCM straight to 8 Ω speaker.
Speaker hang off VUI audio board, not ALSA device on Jetson — no lower-level
path, none needed.

Run TTS **on laptop**, not robot: Kokoro need Python >= 3.10 + current torch,
robot on Python 3.8 w/ JetPack-pinned CUDA 11.4. Laptop has Python 3.12 +
RTX 4060. Pipeline:

```
/g1/tts/text -> kokoro (24 kHz float32) -> resample 16 kHz -> int16 LE
             -> /g1/audio/play_pcm -> PlayStream -> speaker
```

Not implemented yet; when done, replaces `/g1/tts/text` consumption only —
every other topic stays same.

## Known limits

- Replies spoken as one blocking TTS call — no streaming sentence-by-sentence
  playback yet.
- Barge-in handled by ignoring ASR while `/g1/audio/play_state` true — robot
  can't be interrupted mid-sentence except via `/g1/audio/stop`.
- Onboard TTS voice can't change beyond Chinese/English role; see custom
  TTS section above. Demo uses built-in TTS.
- `mic_node` uses plain RMS gate, not real VAD.
- Nothing call locomotion bridge yet; tool-calling from `dialog_node` into
  `g1_control/robot_bridge.py` obvious next step.