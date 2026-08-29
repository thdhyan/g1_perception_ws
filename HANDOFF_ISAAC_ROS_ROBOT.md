# HANDOFF: Isaac ROS on the robot's Orin NX

**Date:** 2026-08-19/20. **Repo:** `g1_perception_ws` on `main`, HEAD `1ac3292`.
**Status at handoff:** the robot's L4T upgrade 35.3.1 → 35.4.1 is **complete and
verified after reboot**. The robot is on JetPack 5.1.2 and ready for the Isaac
ROS `release-2.1` build. Start at Next Steps item 2.

Post-reboot verification (passed):

```
uname -r                -> 5.10.120-tegra
/etc/nv_tegra_release   -> R35 (release), REVISION: 4.1     [JetPack 5.1.2]
eth0                    -> 192.168.123.164/24, driver r8168
wlan0                   -> 10.46.16.14/24, module 8852bu loaded
/usr/local/cuda-11.4    -> unchanged
free -g                 -> 15 total, 13 available
```

The first boot after the upgrade is slow — bootloader and partition updates run
before userspace, so ping and SSH stay dead for a few minutes. That is expected,
not a failure. The rest of the subnet (`.120` Livox LiDAR, `.161` Unitree
controller) stays reachable throughout, which is a quick way to tell a slow boot
apart from a cabling problem.

This is a *different workstream* from [HANDOFF.md](HANDOFF.md), which covers the
sim work and the laptop-side Isaac ROS container. Read this one for anything
touching the robot's onboard Jetson.

The full procedure, with all the measured numbers, lives in
[ISAAC_ROS_ON_ROBOT.md](ISAAC_ROS_ON_ROBOT.md). This document is the state
delta plus what to do next.

## Goal

Run cuVSLAM (`isaac_ros_visual_slam`) and nvblox **on the robot's onboard
Jetson Orin NX 16 GB**, instead of on the laptop, without reflashing the
Jetson.

## Environment facts (measured, not assumed)

| | |
|---|---|
| Robot | `192.168.123.164`, user `unitree`, **passwordless SSH from the laptop**; `sudo` **needs a password** — the human must type it |
| Laptop | `enp2s0` at `192.168.123.222/24`, RTT 0.17 ms |
| Module | Jetson Orin NX 16 GB (reports as "Orin NX Developer Kit"), 15 GB RAM, ~12 GB free |
| OS / ROS | Ubuntu 20.04.6, ROS **Foxy** + Noetic on the host |
| Storage | `/` is a 1.9 TB NVMe, 1.8 TB free. Not a constraint |
| Docker | 24.0.7, `nvidia` runtime present, **default runtime is `runc`** |
| L4T | **was** 35.3.1 (JetPack 5.1.1) → **now** 35.4.1 (JetPack 5.1.2), pending reboot verification |
| CUDA | 11.4, unchanged by the upgrade |
| Unitree services | **none on this Jetson** — only `pulseaudio`. `vui`/loco live on the other controller, so `g1_voice` was never at risk |

## Current progress

1. **Established that current Isaac ROS cannot run here.** It requires JetPack
   7.2. Isaac ROS **`release-2.1`** is the newest release that matches JetPack
   5.x — verified from source, not the docs site: its
   `isaac_ros_common/docker/Dockerfile.aarch64` pins
   `BASE_IMAGE="nvcr.io/nvidia/l4t-base:35.4.1"` and `/usr/local/cuda-11.4`.
   (The docs site renders its platform table in JavaScript and cannot be
   scraped — go to the Dockerfile.)
2. **Upgraded L4T 35.3.1 → 35.4.1 in place**, no reflash, to match that base
   image. Kernel went 5.10.104-tegra → 5.10.120-tegra.
3. **Rebuilt the out-of-tree wifi and bluetooth modules** for the new kernel
   after their DKMS builds failed. Both now show `5.10.120-tegra: installed`.
4. Wrote [ISAAC_ROS_ON_ROBOT.md](ISAAC_ROS_ON_ROBOT.md) — the full procedure,
   memory budget, and gotchas.

## What worked

- **Targeted L4T upgrade instead of `dist-upgrade`:**

  ```bash
  PKGS=$(dpkg -l | awk '/^ii  nvidia-l4t/{print $2}' | tr '\n' ' ')
  sudo apt-get install --only-upgrade $PKGS
  ```

  Simulated first: **0 removals, 40 installs, all `nvidia-l4t-*`**.

- **Fixing the DKMS builds with a symlink.** The Realtek driver Makefiles
  hardcode the *old* kernel path and ignore DKMS's `KVERSION`:

  ```
  make -C /lib/modules/5.10.104-tegra/build M=... modules
  make[1]: *** /lib/modules/5.10.104-tegra/build: No such file or directory.
  ```

  Pointing the stale path at the new headers made both modules build:

  ```bash
  sudo mkdir -p /lib/modules/5.10.104-tegra
  sudo ln -sfn /usr/src/linux-headers-5.10.120-tegra-ubuntu20.04_aarch64/kernel-5.10 \
               /lib/modules/5.10.104-tegra/build
  sudo dkms autoinstall -k 5.10.120-tegra
  ```

  `rtl8852bu` takes 5-20 minutes to compile on this board; `rtkbtusb` is
  seconds. Both ended at `5.10.120-tegra, aarch64: installed`.

- **Answering `N` to the dpkg config-file prompt** for
  `/etc/systemd/nv-oem-config-post.sh`. Unitree deleted that file deliberately;
  restoring NVIDIA's version risks the OEM first-boot wizard running on a
  headless robot. Rule for future prompts: **keep the current version**.

## What didn't work — don't repeat these

- **`sudo apt dist-upgrade`.** Simulated it: wanted to **remove 20 packages**
  (the entire `libopencv-*-dev` set) and drag in an unrelated focal-updates
  upgrade of `libc6`, `udev`, `glib`, `bluez`, `gnupg`. Use the targeted form.
- **Reusing `isaac_ros_ws/src/` from this repo.** Those submodules are pinned
  to **v4.5-0**, which targets JetPack 6.2. They will not build on the robot.
  The robot needs its own `release-2.1` workspace.
- **Trusting `dkms status` as a normal user.** It misreports `rtl8852bu` as
  missing its `dkms.conf`; the file is there, it is a permission artifact. Use
  `sudo dkms status`.
- **Scraping the Isaac ROS docs site** for platform tables. JavaScript-rendered,
  returns only the table of contents. Read the Dockerfiles instead.

## Landmines left behind

- **There is no rollback.** `/boot/Image` was replaced by the upgrade and no
  `Image.backup` was taken; the old kernel's module tree is stripped to just
  the two DKMS modules. Forward is the only direction. `/boot/extlinux/extlinux.conf`
  still has its commented-out `LABEL backup` block if a backup kernel is ever
  staged.
- **`r8169` no longer exists** in 35.4.1 — L4T ships `r8168.ko`. Verified safe:
  the NIC is `[10ec:8168]` and `modules.alias` maps `pci:v000010ECd00008168` to
  `r8168`, so `eth0` should still come up. **This is the SSH lifeline — confirm
  it first after the reboot.**
- **The r35.4 apt source is still installed** at
  `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list`. A future absent-minded
  `apt upgrade` will pull the rest of r35.4 *with* the OpenCV removals. Remove
  or pin it.
- **The DKMS Makefile hardcoding will recur** on every kernel change until the
  Makefiles honour `$(KVERSION)`. The symlink is a workaround, not a fix.

## Next steps

1. ~~**Verify the reboot.**~~ **Done** — see the verification block at the top
   of this document. The robot is on JetPack 5.1.2 with wifi and ethernet up.

2. **Clone and build Isaac ROS 2.1 on the robot** — §5 of
   [ISAAC_ROS_ON_ROBOT.md](ISAAC_ROS_ON_ROBOT.md). Separate workspace at
   `~/workspaces/isaac_ros-dev`, all repos on branch `release-2.1`, then
   `run_dev.sh` and `colcon build --packages-up-to isaac_ros_visual_slam isaac_ros_nvblox`.

3. **Bring up cuVSLAM + nvblox on the D435i** — §6. The D435i has the stereo IR
   pair plus IMU that cuVSLAM wants.

4. **Resolve the ROS distro mismatch before wiring anything to the laptop** —
   §3. The container is **Humble**, the laptop is **Jazzy**, the robot host is
   **Foxy**. Cross-distro DDS is unsupported. Decide up front: keep consumers
   inside the container, bridge deliberately, or run a matching container on
   the laptop. Skipping this turns into a week of silent-topic debugging.

5. **Respect the memory budget** — §7. cuVSLAM + nvblox + a small LLM fits in
   16 GB unified. Adding VoxelNeXt is tight. Cosmos3-Edge does not fit at any
   combination and cannot be quantized (BF16-only).

## Uncommitted work in the repo

```
 M src/g1_isaac_slam/launch/_container_isaac_slam.launch.py
 M src/g1_isaac_slam/launch/isaac_slam.launch.py
?? ISAAC_ROS_ON_ROBOT.md
?? src/g1_isaac_slam/config/g1_sim_isaac.rviz
?? src/g1_voice/
```

`src/g1_voice/` is a **separate, working workstream**: a ROS 2 package wrapping
the G1's VUI/audio service (onboard ASR in, LLM, onboard TTS out). It builds and
its dialog loop was verified end-to-end against the lab Ollama server. It is
unrelated to the Isaac ROS port except that both target the same robot — see
`src/g1_voice/README.md`. None of it is committed yet.
