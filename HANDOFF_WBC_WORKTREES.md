# HANDOFF: WBC / Fall-Recovery Worktree Fleet

> **Other active workstreams:**
> - Sim SLAM/TF/detection → [HANDOFF.md](HANDOFF.md)
> - Isaac ROS / onboard Orin NX → [HANDOFF_ISAAC_ROS_ROBOT.md](HANDOFF_ISAAC_ROS_ROBOT.md) + [ISAAC_ROS_ON_ROBOT.md](ISAAC_ROS_ON_ROBOT.md)
> - Person ReID → [HANDOFF_REID.md](HANDOFF_REID.md)
>
> This document covers the multi-branch WBC / GR00T / fall-recovery effort below.

**Date:** 2026-08-29. **Repo:** `g1_perception_ws`. This work spans **6 branches in
6 separate `git worktree` checkouts**, built up over one session by parallel
subagents. Nothing here is merged to `main` yet — every branch needs review
before merge.

## Goal

Give the G1 humanoid (1) real whole-body control beyond the current
locomotion-only `g1_wbc_node` (arms are currently hard-coded to zero), (2) a
path to NVIDIA's GR00T-WholeBodyControl on real hardware, and (3) fall
recovery that works in **both** supine and prone orientations, in sim as
well as on the real robot (the vendor SDK only covers supine, real-hardware
only).

## Worktree layout

```
~/Projects/thesis/g1_perception_ws                              main
~/Projects/thesis/g1_perception_ws-worktrees/handoff-integration  feat/handoff-integration
~/Projects/thesis/g1_perception_ws-worktrees/groot-wbc-sim         feat/groot-wbc-sim
~/Projects/thesis/g1_perception_ws-worktrees/groot-wbc-real        feat/groot-wbc-real
~/Projects/thesis/g1_perception_ws-worktrees/fall-recovery         feat/fall-recovery
~/Projects/thesis/g1_perception_ws-worktrees/fall-recovery-rl-setup feat/fall-recovery-rl-setup  (branched off feat/fall-recovery)
```

Verify with `git worktree list` from the main worktree. All branches are
currently ahead of `main` only by their own commits — `main` is untouched
except for the one big commit (`f20abe3`) that cleaned up 85 pre-existing
uncommitted changes before this fleet was created.

## ⚠️ Critical gotcha: spawning agents into these worktrees

**Do NOT use the `Agent` tool's `isolation: "worktree"` parameter to work in
these branches.** It creates the harness's *own* sandboxed worktree under
`.claude/worktrees/agent-<id>/`, branched off a stale internal ref — **not**
the `feat/*` branch you asked for, and not even the current `main`. This
was hit twice this session:

- One agent got a worktree at commit `97f979c`, missing `docs/ISAAC_BRINGUP_LAUNCH_PLAN.md`,
  `src/g1_isaac_slam/`, and everything else added since — it correctly
  noticed the mismatch and refused to fabricate work rather than guess.
- Two other agents didn't notice and silently committed real, good work
  onto a throwaway branch named `worktree-agent-<id>` instead of the
  intended `feat/*` branch. Their commits had to be manually recovered:
  `git log worktree-agent-<id> --oneline` to find the commit, then
  `cd` into the *real* `feat/*` worktree and `git cherry-pick <sha>`.

**The working pattern:** spawn the agent with no `isolation` param at all,
and in the prompt tell it the exact absolute path of the real worktree
(e.g. `/home/thakk100/Projects/thesis/g1_perception_ws-worktrees/groot-wbc-sim`,
branch `feat/groot-wbc-sim`, already created) and instruct it to `cd` there
for every command and never touch sibling worktree dirs. This worked
cleanly for the arm-IK and RL-training-setup agents. If you ever do end up
with orphaned work on a `worktree-agent-*` branch, the cherry-pick recovery
above is safe and was used twice without incident.

Also note: `.claude/worktrees/agent-*` dirs left behind by past isolated
runs (`agent-a35841ce3a8d9e9dd`, `agent-a999c343843fa0f97` as of this
writing) are safe to ignore/delete once their commits are confirmed
cherry-picked onto the right branch — check with `git branch --contains`
before deleting anything.

## Per-branch status

### `feat/handoff-integration` — BLOCKED, no commits

**Goal:** integrate github.com/lzyang2000/HANDOFF (arXiv:2606.06493), a
distilled MoE policy bundling WBC + locomotion + fall-recovery, native G1
target, ROS2 topic interface (`/g1/command`, 18-float, controller node →
`sim_policy_node.py`/`hardware_node.py`).

**What didn't work:** the repo has **no license anywhere** — confirmed via
README, `pyproject.toml`, and `GET /repos/lzyang2000/HANDOFF` returning
`"license": null`. Under default copyright this blocks any vendoring,
submoduling, or derivative code. The assigned agent correctly stopped here
rather than write integration code against unlicensed source.

**Side finding:** this repo already has a home-grown, similarly-named
`src/g1_wbc/g1_wbc/wbc_node.py` ("Sim-agnostic GR00T Whole-Body Control
node") — read that before assuming HANDOFF is still needed even if the
license gets resolved later.

**Next steps:**
1. Contact the HANDOFF author about licensing, or re-check the repo for a
   license added since 2026-08-29 (`git ls-remote`/re-clone).
2. If unblocked: decide whether HANDOFF replaces or complements the
   existing `g1_wbc_node`, given the licensing delay already made
   `feat/groot-wbc-sim` progress independently on the arm gap (see below).
3. If still blocked indefinitely: consider this branch dead, document why
   in a final commit, and skip it.

### `feat/groot-wbc-sim` — working, sim-verified (commits `ff9a34f`, `161bed7`, `1e1cf33`)

**Goal:** upper-body arm control for `g1_wbc_node` (which hard-codes all 14
arm joints to 0), informed by NVIDIA's GR00T-WholeBodyControl's decoupled
upper/lower architecture.

**What worked:**
- License check on github.com/NVlabs/GR00T-WholeBodyControl: dual
  Apache-2.0 (source, safe to read/adapt) + NVIDIA Open Model License
  (pretrained weights only — re-read that text before using any checkpoint).
- Their actual lower-body RL policy was judged **not worth porting** —
  ZMQ/Docker/MuJoCo-native, no ROS2 bridge, different physics than this
  repo's Gazebo Harmonic sim. Only the *architectural idea* (decoupled
  upper/lower, independently toggled) was reused.
- Implemented real forward/inverse kinematics from scratch in
  `src/g1_wbc/g1_wbc/arm_kinematics.py`: parses the G1 URDF's torso→hand
  chain with `urdf_parser_py` (no `pinocchio`/`kdl_parser_py`/`ikpy`
  available in `.venv` — none were added), FK as chained numpy homogeneous
  transforms, IK via damped-least-squares Jacobian with a numerical
  (finite-difference) Jacobian. **16/16 unit tests pass**,
  sub-millimeter position error, sub-degree orientation error.
- `arm_ik_node.py` subscribes `/g1/left_arm_target_pose` and
  `/g1/right_arm_target_pose` (`geometry_msgs/PoseStamped`, `torso_link`
  frame — split from a single ambiguous topic in the original stub) and
  publishes `std_msgs/Float64` on `/g1/joint/<arm_joint_name>` at 30Hz,
  matching `wbc_node.py`'s existing publish pattern exactly.
- Added a `control_arms` param to `wbc_node.py` (default `true`) so it
  stops publishing arm joints when `enable_arm_ik:=true` is passed to
  `sim.launch.py` / `wbc.launch.py` — verified in a **real headless Gazebo
  run** that the two nodes don't fight (stable IK-solved values, no
  flapping back to the old zero-hold).

**What didn't work / open caveat:** `/joint_states` wasn't ticking at all
during the headless test run (`ros2 topic hz` → 0Hz) — looked like a
pre-existing sim/physics-plugin quirk unrelated to this work, **not
independently confirmed** that the Gazebo model's physical joints actually
moved, only that the `/g1/joint/*` command topics (the actual IK contract)
carried correct values. Chase this down before trusting visual
confirmation in Gazebo.

**Next steps:**
1. Investigate the `/joint_states` 0Hz issue — needed for any visual/E2E
   confirmation of arm motion, and for anything that depends on joint
   feedback (e.g. closed-loop grasping later).
2. Decide a real source for `/g1/{left,right}_arm_target_pose` — right now
   nothing publishes to it in this repo (this branch only proved the
   solver+node work when driven by hand via `ros2 topic pub`). Candidates:
   `human_selector_node.py`'s `/g1/selected_human` for a reach/greet
   gesture, or a teleop source.
3. Merge review: this is the most finished branch — good candidate to
   review/merge to `main` first, once the `/joint_states` question is
   resolved or explicitly deferred.

### `feat/groot-wbc-real` — plan + non-executing stubs only (commit `5c9b20e`)

**Goal:** real-hardware deployment plan for GR00T-WholeBodyControl.

**What worked:** thorough recon, explicitly **not** attempted to run
anything on hardware (correct — this is safety-critical, real robot).
- License: same Apache-2.0 + NVIDIA Open Model License as above.
- **Key architectural finding:** GR00T-WBC's control loop is a *replacement*
  locomotion controller (50Hz joint-level), not an add-on to the
  `LocoClient` high-level API `robot_bridge.py` already uses — the two
  **cannot hold the robot's joints concurrently**. Any real integration
  must gate on "only one controller owns the joints," checked via
  `robot_bridge.py`'s `get_fsm` before a WBC session starts.
- Onboard deployment needs JetPack 6.1+ (a full reflash); this robot is on
  5.1.2, kept there specifically to avoid a reflash for existing Isaac ROS
  work (see [HANDOFF_ISAAC_ROS_ROBOT.md](HANDOFF_ISAAC_ROS_ROBOT.md)).
  Plan recommends the off-board (laptop) deploy path instead — **but this
  is unconfirmed by upstream docs for the full 50Hz loop**, two open
  upstream GitHub issues (#147, #149) ask this exact question with no
  maintainer answer as of 2026-08-29.
- This workspace's `SPACE` e-stop (hardened: bypasses dispatch lock,
  threaded so it can't be blocked) is confirmed stronger than GR00T-WBC's
  documented `O`-key "exit control safely" — do not trust that phrase,
  measure real stop latency on hardware before any tethered test.
- Also caught a pre-existing, unrelated doc bug: `REAL_ROBOT_WORKFLOW.md`
  says the robot's IP is `.220`, `RUNBOOK_REAL_ROBOT.md` says `.222` —
  needs a human to check which is actually correct and fix the wrong doc.

**Next steps (explicit human-required list from the plan, §8):**
1. A human decides the JetPack 6 reflash question (affects other active
   workstreams — do not decide this in an agent).
2. Read the NVIDIA Open Model License text directly before using any
   checkpoint (not yet done beyond the summary above).
3. Resolve the off-board-vs-onboard 50Hz-loop feasibility question,
   ideally by asking upstream directly (issues #147/#149) rather than
   guessing.
4. Reconcile the `.220` vs `.222` doc discrepancy.
5. Only after 1–4: measure `O`-key stop latency on real hardware, verify
   exact TensorRT version pin (upstream calls a mismatch a cause of
   "dangerous robot behavior"), then phase in bench → tethered →
   free-standing tests with a human physically present and E-stop in hand
   at every phase. Do not skip phases.

**This branch should stay plan-only until a human clears item 1.** Do not
spawn an agent to "finish" this one the way `groot-wbc-sim` was finished —
it is not a sim-testable stub, it is real-hardware scaffolding.

### `feat/fall-recovery` — real-robot dispatch done+tested; sim demo scaffolds but doesn't stand (commits `90e5781`..`7d13378`)

**Goal:** fall recovery in both supine and prone, real robot and sim.

**What worked (real-robot side, commits `800b3da`, `5bcd60b`, `6495a06`):**
- Vendor SDK reality check: `robot_bridge.py` already owns a
  `unitree_sdk2py...LocoClient` instance, which ships a built-in
  `Lie2StandUp()`. **Confirmed via Unitree's own G1 docs
  (docs.quadruped.de) that this is supine-only, flat/hard-ground-only,
  real-hardware-only** — controller instructions literally say "place G1
  facing up" first. (An earlier web search result conflated this with the
  *HumanUP research paper's* prone+supine results — that's a different,
  unreleased RL policy, not the SDK call — don't repeat that mix-up.)
- Added `{"cmd": "recover"}` → `Damp()` → sleep → `Lie2StandUp()` dispatch
  case in `robot_bridge.py`, matching the existing `damp`/`move` pattern
  exactly. Triggered via a new `U` key in the teleop console (the plan's
  proposed `R` was already taken by "rescan"). `SPACE` e-stop semantics
  untouched — recovery is a deliberate separate action, never
  auto-triggered.
- Mock-based unit test (`src/g1_control/test/test_recover.py`), **6/6
  pass**, confirmed zero real-hardware calls made during testing (no
  `ChannelFactoryInitialize()`, no DDS participant ever created).

**What worked (sim side, scaffolding only, commits `9ed09fc`, `fcbad21`, `7d13378`):**
- Added `/g1/control_mode` (`std_msgs/String`, `"wbc"`/`"recovery"`) to
  `wbc_node.py` as a joint-ownership arbitration mechanism — it skips its
  own publish when mode isn't `"wbc"`. `get_up_node.py` claims/releases it
  correctly on both ends, verified in a real Gazebo run.
- Two `std_srvs/Trigger` services, `/g1/recover_sim_supine` and
  `/g1/recover_sim_prone`, wired into `wbc.launch.py` behind
  `enable_recovery` (default off).
- Real Gazebo test (headless): spawned the robot lying via
  `/world/warehouse/set_pose`, triggered each service. Both ran to
  completion, both returned `success: true`, mode handoff back to `wbc`
  worked both times, joints visibly tracked the commanded keyframes.

**What didn't work:**
- HumanUP (github.com/RunpeiDong/humanup, Apache-2.0) ships **no G1
  checkpoint and no directly-replayable joint-trajectory dataset** — only
  Isaac Gym training code plus an internal-format "discovered
  trajectories" bootstrap folder, not usable as open-loop playback without
  more work than this pass allowed.
- The keyframe sequences actually used (tuck→roll→push-to-kneel→stand for
  supine; hands-under-shoulders→hands-and-knees→kneel→stand for prone) are
  **hand-authored**, sanity-bounded only against URDF joint limits — not
  motion-captured, not policy-derived. **The robot does not stand up** in
  either orientation: pelvis height peaked ~0.39m (supine) and stayed
  ~0.06–0.26m (prone) vs. ~0.74m standing. This is expected for a
  hand-authored open-loop sequence with no contact-dynamics feedback —
  don't be surprised it doesn't work, that's exactly the gap
  `feat/fall-recovery-rl-setup` exists to close.
- A real cross-agent hazard was found and worked around: **Gazebo
  Transport discovery is not scoped by `ROS_DOMAIN_ID`** — running two
  `gz sim` instances with the same world name from different worktree
  agents made them mutually visible over the network. Fixed with
  `GZ_PARTITION` env var isolation. **If you run sim tests from multiple
  worktrees/agents concurrently again, set a unique `GZ_PARTITION` per
  agent up front** — don't rediscover this the hard way.

**Next steps:**
1. Either retune the hand-authored keyframes against real contact dynamics
   (uncertain payoff, iterative), or treat this branch's sim demo as
   "scaffold proven, behavior pending real RL policy" and prioritize
   `feat/fall-recovery-rl-setup` instead.
2. Once an RL-trained policy exists (see below), `get_up_node.py` needs an
   ONNX-inference path added (currently pure scripted playback, no policy
   loading) — check `wbc_node.py`'s existing ONNX consumption pattern for
   the contract to match.
3. `robot_bridge.py`'s real-robot `recover` dispatch is done and tested at
   the mock level — it has never been run against real or simulated
   hardware. First real-hardware test needs a human present, per this
   repo's existing real-robot safety conventions
   ([RUNBOOK_REAL_ROBOT.md](RUNBOOK_REAL_ROBOT.md)).

### `feat/fall-recovery-rl-setup` — pipeline documented, not launched (commit `e0322e4`)

**Goal:** prep a full HumanUP-based RL training pipeline (both
orientations) for later launch on a bigger GPU — this machine's GPU (RTX
4060 Laptop, 8GB, ~3.6GB free at time of testing) can't fit HumanUP's
default 4096-parallel-env training.

**What worked:**
- Confirmed **both orientations are natively supported by HumanUP's own
  config system**, no new curriculum authoring needed:
  `g1waist`/`g1track` = supine (`facingup_poses.npy`),
  `g1waistroll`/`g1rolltrack` = prone (`facingdown_poses.npy`,
  `init_state.rot=[0,-0.707,0,0.707]`).
- Exact install + launch commands for Stage I → Stage II → `save_jit.py`,
  for both orientations, documented in
  `FALL_RECOVERY_RL_TRAINING_PLAN.md` in that worktree.
- Vendored the relevant config files (not the full repo, not large
  data blobs) under `src/g1_wbc/rl_training/humanup/`, with
  `ATTRIBUTION.md`.
- `--num_envs` is a CLI override already, not a new config — a documented
  ~256–512-env "debug/pipeline-check tier" exists for this GPU, explicitly
  **not** claimed to produce deployable quality at that scale.

**What didn't work / blockers:**
- **Isaac Gym is not installed and not pip-installable** — NVIDIA requires
  a manual, account-gated download (Preview 4) from their site. Nothing
  was installed this pass.
- **Real mismatch found, not yet resolved:** HumanUP's G1 policy is
  23-action / 868-dim-obs (legs+waist+arms). `wbc_node.py`'s current
  balance/walk policies are 15-action / 516-dim-obs (legs+waist only, arms
  held at zero). A future `"recover"` inference mode in `wbc_node.py` (or
  `get_up_node.py`) will need **its own observation-builder** and will
  need to actually publish arm joint targets — this was left as a
  documented checklist item, not implemented.

**Next steps (needs a human decision on hardware/budget, not an agent):**
1. Get access to a bigger GPU (cloud, or a lab machine with 16GB+ VRAM).
2. Manually download Isaac Gym Preview 4 (account-gated, can't be scripted
   headlessly) and install per `FALL_RECOVERY_RL_TRAINING_PLAN.md`.
3. Launch Stage I then Stage II for both orientations per that doc's exact
   commands. This is a multi-day unattended job — expect to check in
   periodically, not babysit it in one session.
4. Export via `save_jit.py`, then patch the JIT export to ONNX (small
   patch, documented) and write the obs-builder mismatch fix from the
   blocker above before wiring the trained policy into `get_up_node.py`.

## General notes for whoever picks this up

- All heavy install/build work (Isaac Gym, colcon builds) should be run in
  the **main session, not a subagent** — matches this repo's existing
  `AGENTS.md` rule about `VoxelKP`'s `setup.py develop` crashing subagent
  processes; the same caution class applies here.
- None of these 6 branches are merged. Recommend reviewing and merging
  `feat/groot-wbc-sim` first (most complete, sim-verified), then
  `feat/fall-recovery`'s real-robot half (tested, low-risk, additive-only
  dispatch case) separately from its sim half (scaffold-only, still needs
  the RL policy to be meaningful).
- `feat/handoff-integration` and `feat/groot-wbc-real` should stay
  unmerged/paused until their respective blockers (license; human
  JetPack/network/license decisions) clear.
