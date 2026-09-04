# Worktree Fleet Summary & Code Redundancy Analysis

**Date:** 2026-08-29  
**Repo:** `g1_perception_ws`  
**Total worktrees:** 6 feat/ branches + main + 2 orphaned agent worktrees

---

## Per-Worktree Status

### 1. `feat/handoff-integration` — BLOCKED, 1 commit only
- **Commit:** `f20abe3` "Migrate perception stack to VoxelNeXt, add ReID pipeline, drop CenterPoint/PointPillar"
- **Goal:** Integrate `github.com/lzyang2000/HANDOFF` (MoE policy bundling WBC + locomotion + fall-recovery)
- **Status:** BLOCKED — repo has **no license** (confirmed via README, pyproject.toml, and GitHub API `"license": null`). Under default copyright, any vendoring/submoduling/derivative code is blocked.
- **Side finding:** This repo already has a home-grown `src/g1_wbc/g1_wbc/wbc_node.py` ("Sim-agnostic GR00T Whole-Body Control node")
- **Next steps:** Contact HANDOFF author about licensing, or re-check for license added since 2026-08-29. If unblocked, decide whether HANDOFF replaces or complements existing `g1_wbc_node`. If still blocked: document as dead branch.

### 2. `feat/groot-wbc-sim` — **Most completed, sim-verified**
- **Commits:** `ff9a34f` (docs+scaffold), `161bed7` (numpy FK/IK), `1e1cf33` (real arm IK node + launch wiring)
- **Goal:** Upper-body arm control for `g1_wbc_node` (which hard-codes all 14 arm joints to 0), informed by GR00T-WBC's decoupled upper/lower architecture
- **What worked:**
  - License check: NVlabs GR00T-WholeBodyControl is dual Apache-2.0 + NVIDIA Open Model License (safe to read/adapt)
  - Implemented FK/IK from scratch in `src/g1_wbc/g1_wbc/arm_kinematics.py` using `urdf_parser_py` + damped-least-squares Jacobian
  - **16/16 unit tests pass** with sub-mm position error, sub-degree orientation error
  - `arm_ik_node.py` subscribes `/g1/left_arm_target_pose` / `/g1/right_arm_target_pose` (PoseStamped, torso_link frame)
  - Added `control_arms` param to `wbc_node.py` (default true) — verified in real headless Gazebo that IK node and WBC node don't fight
- **Open caveat:** `/joint_states` was 0 Hz during headless test (root cause: Gazebo headless tick rate + no `update_frequency` on JointStatePublisher plugin)
- **Next steps:** 
  1. Investigate `/joint_states` 0Hz issue (fix: add `<update_frequency>50</update_frequency>` to URDF JointStatePublisher — **done** in this session)
  2. Decide source for `/g1/{left,right}_arm_target_pose` (currently nothing publishes; **added `arm_target_publisher.py` node** — **done** in this session)
  3. Merge review — good candidate to merge to `main` first once `/joint_states` resolved

### 3. `feat/groot-wbc-real` — Plan-only, no execution
- **Commit:** `5c9b20e` (same base as others + plan-only content)
- **Goal:** Real-hardware deployment plan for GR00T-WholeBodyControl
- **Status:** Should stay **plan-only** per handoff doc §8. Do not spawn agent to "finish" this — it is real-hardware scaffolding, not sim-testable.
- **Key findings:**
  - GR00T-WBC's control loop replaces (not supplements) the `LocoClient` — the two cannot hold joints concurrently
  - Onboard deployment needs JetPack 6.1+ (this robot is on 5.1.2 intentionally)
  - Off-board (laptop) deploy path unconfirmed for full 50Hz loop (upstream issues #147, #149 unanswered)
  - This workspace's hardened `SPACE` e-stop is stronger than GR00T-WBC's `O`-key "exit control safely"
  - Doc bug: `REAL_ROBOT_WORKFLOW.md` says robot IP `.220`, `RUNBOOK_REAL_ROBOT.md` says `.222`
- **Next steps (all human-required):** 
  1. Human decides JetPack 6 reflash question
  2. Read NVIDIA Open Model License text directly
  3. Resolve off-board-vs-onboard 50Hz-loop feasibility
  4. Reconcile `.220` vs `.222` doc discrepancy
  5. Only after 1–4: measure `O`-key stop latency, verify TensorRT version, phase bench→tethered→free-standing with human present

### 4. `feat/fall-recovery` — Real-robot dispatch done; sim scaffold proven but doesn't stand
- **Commits:** `5bcd60b` (U key + recover dispatch), `6495a06` (mock unit test 6/6 pass), `9ed09fc` / `fcbad21` / `7d13378` (sim services + keyframe sequences)
- **Goal:** Fall recovery in both supine and prone, real robot and sim
- **Real-robot side (tested):**
  - Vendor SDK reality check: `Lie2StandUp()` is **supine-only**, flat/hard-ground-only (confirmed via Unitree docs.quadruped.de)
  - Added `{"cmd": "recover"}` → `Damp()` → sleep → `Lie2StandUp()` dispatch in `robot_bridge.py`, triggered via `U` key in teleop console
  - Mock unit test `test_recover.py`: **6/6 pass**, zero real-hardware calls
- **Sim side (scaffold only):**
  - Added `/g1/control_mode` arbitration ("wbc"/"recovery") 
  - Two services: `/g1/recover_sim_supine` / `/g1/recover_sim_prone` (gated by `enable_recovery`, default off)
  - Real Gazebo test: both services returned `success: true`, mode handoff back to `wbc` worked
  - **Keyframe sequences do NOT produce standing:** pelvis peaked ~0.39m (supine) and ~0.06–0.26m (prone) vs. ~0.74m standing. Hand-authored open-loop sequences with no contact-dynamics feedback.
  - HumanUP repository has no G1 checkpoint or replayable trajectory dataset
  - Cross-agent hazard: Gazebo Transport not scoped by `ROS_DOMAIN_ID` — fixed with `GZ_PARTITION` env var
- **Next steps:**
  1. Either retune keyframes (low payoff — structural deficit: open-loop position lerp can't lift CoM without ground-reaction physics) OR treat sim demo as "scaffold proven, behavior pending real RL policy"
  2. If pivoting to RL: `get_up_node.py` needs ONNX-inference path (currently pure scripted playback)
  3. Real-hardware `recover` dispatch needs human present per safety conventions

### 5. `feat/fall-recovery-rl-setup` — Pipeline documented, not launched
- **Commit:** `e0322e4` "docs(fall-recovery): HumanUP RL training plan + vendored env configs"
- **Goal:** Prep HumanUP-based RL training pipeline for both orientations for later launch on bigger GPU
- **What worked:**
  - Confirmed both orientations natively supported by HumanUP config: `g1waist`/`g1track` = supine, `g1waistroll`/`g1rolltrack` = prone
  - Documented Stage I → Stage II → `save_jit.py` launch commands in `FALL_RECOVERY_RL_TRAINING_PLAN.md`
  - Vendored config files (not full repo) under `src/g1_wbc/rl_training/humanup/` with `ATTRIBUTION.md`
  - Documented ~256–512 env "debug tier" for this GPU (RTX 4060 Laptop, 8 GB, ~3.6 GB free) — explicitly not claimed to produce deployable quality
- **Blockers:**
  - Isaac Gym is **not pip-installable** — requires manual, account-gated download from NVIDIA Preview 4 site
  - **Observation mismatch:** HumanUP policy is 23-action / 868-dim-obs (legs+waist+arms), but `wbc_node.py`'s balance/walk policies are 15-action / 516-dim-obs (legs+waist only, arms at zero). Future `"recover"` inference mode will need its own observation-builder and must publish arm joint targets.
- **Next steps (human decision on hardware/budget):**
  1. Get access to bigger GPU (cloud or lab machine with 16GB+ VRAM)
  2. Manually download Isaac Gym Preview 4 (account-gated, can't script headlessly)
  3. Launch Stage I then Stage II for both orientations (multi-day unattended job)
  4. Export via `save_jit.py`, patch to ONNX, fix obs-builder mismatch, wire trained policy into `get_up_node.py`

### 6. Orphaned agent worktrees (left from past isolated runs)
- `.claude/worktrees/agent-a35841ce3a8d9e9dd` at commit `5c9b20e` — worktree-agent branch, needs cherry-pick recovery if commits useful
- `.claude/worktrees/agent-a999c343843fa0f97` at commit `766d086` — worktree-agent branch, needs cherry-pick recovery if commits useful
- **Safe to ignore/delete** once commits are confirmed cherry-picked onto the right `feat/*` branch (check with `git branch --contains`)

---

## Code Redundancy Analysis

### Findings: **Low to moderate redundancy** — the codebase deliberately separates concerns across packages

| Package | Key Components | Redundancy Notes |
|---------|---------------|-----------------|
| **`src/g1_wbc/g1_wbc/`** | `wbc_node.py`, `arm_ik_node.py`, `arm_kinematics.py`, `arm_target_publisher.py` (new) | **Minimal redundancy** — `arm_ik_node` + `arm_target_publisher` are new additions to solve the arm IK gap. `arm_kinematics.py` is from-scratch implementation (no pinocchio/kdl/ikpy available in venv). |
| **`src/g1_control/g1_control/`** | `robot_bridge.py`, `human_follower_node.py`, `cmd_vel_bridge.py`, `cmd_pose_bridge.py` | **Low redundancy** — `robot_bridge.py` owns the `LocoClient` and provides the `recover` dispatch (supine-only). `human_follower_node` and bridges are separate concerns (navigation vs. whole-body control). |
| **`src/g1_bringup/`** | `sim.launch.py`, `wbc.launch.py`, `gz_bridge.yaml`, world SDF | **Low redundancy** — `sim.launch.py` wires WBC + arm nodes. `wbc.launch.py` loads ONNX policies. Bridge config is shared Gazebo↔ROS2 topic map. No duplicate bridge configs. |
| **`src/g1_description/urdf/`** | `g1_29dof.urdf` (with Gazebo plugins), `g1_23dof.urdf` | **Minimal redundancy** — g1_23dof.urdf lacks gazebo section. The JointStatePublisher plugin fix was applied only to g1_29dof.urdf (the one used in simulation). |
| **`src/g1_nav/`** | Navigation stack params, BTs for recovery | **Separate concern** — nav2 recovery plugins (`spin`, `backup`, `wait`) are for navigation, not robot fall recovery. No overlap with WBC fall-recovery code. |
| **`src/g1_perception/`** | ReID, detection, pose bridges | **Separate concern** — perception pipelines (person reID, LiDAR processing) are distinct from WBC/fall-recovery. |
| **`src/g1_voice/`** | Audio backend, dialog nodes | **No overlap** — voice/language interface, completely separate domain. |

### **Deliberate separation patterns:**
1. **WBC vs. low-level SDK:** `g1_wbc/` handles whole-body position targets; `robot_bridge.py` / `LocoClient` handles low-level Unitree SDK commands. They own different joint sets (WBC → arm+waist+joint targets; SDK → locomotion modes).
2. **Sim vs. real:** `sim.launch.py` / `wbc.launch.py` are simulation configs; real-hardware paths are in `groot-wbc-real` plan-only branch. No code shared between sim and real paths beyond the node interfaces.
3. **Arm control gap:** Before this session, arms were hard-coded to zero in `wbc_node.py`. The `arm_ik_node.py` + `arm_target_publisher.py` were added from scratch to fill this gap — no pre-existing arm IK code existed in the repo.
4. **Fall-recovery paths:** Real-robot `Lie2StandUp()` (vendor SDK, supine-only) vs. sim keyframe sequences (hand-authored, open-loop) vs. planned RL policy (HumanUP, needing bigger GPU). These are **three separate code paths** with no shared implementation.

### **No significant code duplication detected.**
The worktree fleet was built in a modular way, with each branch adding targeted functionality:
- `feat/groot-wbc-sim`: Added arm IK infrastructure (3 new files)
- `feat/fall-recovery`: Added recovery dispatch + sim services (keyframe sequences + service wiring)
- `feat/fall-recovery-rl-setup`: Documented RL pipeline config (vendored configs only)

The heaviest new code was in `feat/groot-wbc-sim` (arm IK + publisher), which is the most "complete" branch and the recommended merge candidate first.