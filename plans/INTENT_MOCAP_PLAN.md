# Plan: PhaseSpace-labelled scene → intent-sim LLM plan → G1 goto (LocoClient)

Status: BUILT + dry-run verified 2026-10-07 (see RUNBOOK_INTENT.md). Not yet run on the real robot/PhaseSpace. Decisions confirmed by user:
- LLM provider/model are ROS params; any intent-sim provider (openai, openrouter, nim, gemini,
  ollama/dl/spark0x). Default `provider=openai`, `model=gpt-6.1-sol` (verify id against the API).
- Motion: LLM gives an intent-sim pose-goal trace; goto loop closes on PhaseSpace pose, never odom.
- Orientation: captured in the labeler GUI from per-LED TFs (Kabsch) + typed heading.
- Terminal call: stop, then arm gesture (talk/low_wave → wave, shake_hand → shake_hand, listen/stand_still → none).

## Data flow

```
phasespace_ros2 tracker ─TF phasespace→<hat>, <hat>/led_<id>, /phasespace/groups
        │
g1_mocap_labeler/labeler_gui  (GUI 1: name, role human|robot, heading, capture)
        ├─ TF phasespace → <name>      (robot: phasespace → g1_mocap)
        ├─ /mocap/agents  String JSON  {frame, agents:[{name, role, x, y, yaw_deg, oriented, age}]}
        └─ ~/.ros/g1_mocap_labels.yaml (labels + LED references, reloaded next session)
        │
g1_intent/intent_planner  ← /intent/trigger (service, task text)
        ├─ snapshot /mocap/agents → intent-sim Scene (canvas y-down; offset/flip recorded)
        ├─ render_system_prompt(prompt_style=proxemics) → LLMClient(provider, model)
        ├─ multi-round static loop against the intent-sim 2D sim → trace.json
        └─ /intent/plan  String JSON (calls + pose goals in phasespace frame), run dir runs_intent/<id>/
        │
g1_intent/intent_review_gui  (GUI 2: intent-sim-style top-down render, call list,
        │                     velocity preview; Enter approve / R replan / Esc reject / Space E-STOP;
        │                     live robot overlay during execution)
        ▼  /intent/approve
g1_intent/goto_executor  (port of G1_sim/g1_sim/goto_trace.py controller)
        ├─ pose = robot mocap TF (stop if older than 0.3 s)
        ├─ set_velocity → robot_bridge.py socket (LocoClient.SetVelocity), ~10 Hz, deadman
        ├─ terminal call → stop + arm gesture via robot_bridge
        └─ execution_report.json
```

## Deliverables
- `src/g1_mocap_labeler` (1 node, GUI 1)
- `src/g1_intent` (3 nodes: intent_planner, intent_review_gui, goto_executor; 1 launch)
- `RUNBOOK_INTENT.md` (build, terminal order, checks, E-stop)
- Dry-run mode: executor `dry_run:=true` (no bridge) + fake mocap publisher for testing without the lab.

## Verify
1. Build both packages with colcon.
2. Fake `/phasespace/groups` + TFs → label in GUI 1 → `/mocap/agents` correct.
3. Trigger with a stub/cheap model → trace.json + GUI 2 render (screenshot checked).
4. Executor dry_run against a simulated kinematic robot reaches all goals.
5. Real robot: run the runbook with the robot on the gantry first.

## Status (2026-10-07)

Verified on the laptop:
- **Dry run:** `fake_mocap:=true dry_run:=true`, with the review GUI and executor running headless under Xvfb.
- **Labeller:** LED-fit yaw is within 0.5°.
- **Planning:**
  - openai `gpt-6.1-sol` planned in 7 s.
  - openrouter `anthropic/claude-sonnet-4.5` planned in 11 s.
- **Execution:**
  - Goals were reached closed-loop on the fake PhaseSpace (pos err < 0.15 m, yaw err < 4°).
  - The wave gesture path ran (dry).
  - `execution_report.json` and `review.mp4` were written.
- **Bag replay:** `use_sim_time:=true` reaches all nodes.

Fixes found while testing:
- The system `python3-brotli` broke intent-sim's httpx inside the ROS env. `scene_bridge._hide_old_brotli` works around it.
- rclpy forbids one log call site from using several severities.

Not done:
- A real-robot run.
- Live data collection: the robot and PhaseSpace were unreachable. Use `scripts/record_intent_session.sh`.
- Disk space: `/` has 18 GB free and `/Storage` is not mounted.
