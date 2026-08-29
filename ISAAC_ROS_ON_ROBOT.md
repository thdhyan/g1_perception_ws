# Isaac ROS on the G1's Orin NX — without reflashing

Target: run cuVSLAM (`isaac_ros_visual_slam`) and nvblox on the robot's onboard
Jetson Orin NX 16 GB, keeping Unitree's own services (`vui`, loco, sensor
drivers) intact.

**The rule this whole document exists to protect: do not reflash the robot's
Jetson.** There is no published Unitree restore image, and the sensor stack
that [REAL_ROBOT_WORKFLOW.md](REAL_ROBOT_WORKFLOW.md) depends on lives here.

Measured, not assumed: this Jetson runs **no Unitree services** — a process
scan found only `pulseaudio`. `vui`, loco, and the arm services run on the
robot's other controller, so `g1_voice` and `robot_bridge` are not at risk from
work done here. That lowers the stakes but does not remove them.

---

## 0. Ethernet link

Working as of the last check: laptop `enp2s0` up at `192.168.123.222/24`,
robot answering at `192.168.123.164` in 0.17 ms, passwordless SSH as `unitree`.

If the link is down again, `NO-CARRIER` means no cable link — not an IP problem:

```bash
sudo ip link set enp2s0 up
ip -br link show enp2s0          # want UP, not NO-CARRIER
sudo ip addr add 192.168.123.222/24 dev enp2s0
ping -c3 192.168.123.164
```

---

## 1. Preflight — read the robot's actual state

Everything below branches on these five answers. Run on the robot:

```bash
cat /etc/nv_tegra_release          # L4T version -> JetPack version
nvcc --version 2>/dev/null; ls -d /usr/local/cuda-*   # CUDA
free -g                            # unified RAM (expect 16)
df -h / /home; lsblk               # free space, is there an NVMe?
docker --version; docker info 2>/dev/null | grep -i runtime   # nvidia runtime present?
systemctl list-units --type=service | grep -iE "unitree|vui|voice|loco"  # what must not break
```

`/etc/nv_tegra_release` prints something like `# R35 (release), REVISION: 4.1`.
Read it as L4T 35.4.1.

### Which Isaac ROS release to use

| L4T on the robot | JetPack | Isaac ROS release | Reflash? |
|---|---|---|---|
| 35.4.1 | 5.1.2 | **release-2.1** | no — this is the happy path |
| 35.3.1 | 5.1.1 | release-2.0, or in-place apt bump to 5.1.2 then 2.1 | no |
| 36.x | 6.x | release-3.x / 4.x depending on minor | no |
| anything else | — | see §8 | — |

Verified from source, not from the docs site: `isaac_ros_common` on branch
`release-2.1` sets `BASE_IMAGE="nvcr.io/nvidia/l4t-base:35.4.1"` and
`/usr/local/cuda-11.4`. The docs site renders its platform table in JavaScript
and cannot be scraped, so the Dockerfile is the authority here.

---

## 2. What you cannot reuse

`isaac_ros_ws/src/` in this workspace is pinned to **v4.5-0**, which targets
JetPack 6.2. It will not build or run on a JetPack 5 robot. Do not try to
force it — clone a separate, release-2.1 workspace on the robot instead.

Also note what you give up by going back to 2.1: it predates the newer cuVSLAM
releases, so no multi-camera cuVSLAM and none of the recent nvblox performance
work. It is a working SLAM stack, not the current one.

---

## 3. ROS distro reality check — read before you invest a day

Isaac ROS 2.1 containers ship **ROS 2 Humble** inside. That part is fine: the
robot host runs Foxy, but the container has its own ROS install and never
touches it.

The problem is the laptop, which runs **Jazzy**. Cross-distro DDS interop
(Humble ↔ Jazzy, and Foxy ↔ anything) is not supported and message hashes
differ. Plan for one of these up front:

- **Keep consumers inside the container.** Run cuVSLAM, nvblox, and anything
  that eats their output in the same Humble container. Publish only a thin,
  stable result to the laptop.
- **Bridge deliberately.** A Foxglove or Zenoh bridge, or a `ros2 bag` handoff,
  rather than hoping raw DDS interops.
- **Match distros.** Run a Humble container on the laptop for the parts that
  need to talk to the robot's Isaac ROS.

Do not skip this; it is the most common way this port turns into a week of
confusing silent-topic debugging.

---

## 4. Storage

Isaac ROS wants **30+ GB** for images and build artifacts. Not a problem here:
`/` is a **1.9 TB NVMe with 1.8 TB free**, and Docker's root is already on it
at `/var/lib/docker`. Nothing to move, nothing to plan around.

---

## 5. Setup

On the robot, all inside its own directory so nothing collides with Unitree's
files:

```bash
sudo apt update && sudo apt install -y git-lfs curl
git lfs install --skip-repo

mkdir -p ~/workspaces/isaac_ros-dev/src && cd ~/workspaces/isaac_ros-dev/src
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git

echo "export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev" >> ~/.bashrc
source ~/.bashrc
```

JetPack already installs Docker and the NVIDIA container runtime; confirm
rather than reinstalling:

```bash
docker info | grep -i "default runtime"      # want: nvidia
```

Enter the dev container (first run builds the image — slow, and this is where
the 30 GB goes):

```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && ./scripts/run_dev.sh
```

Inside the container:

```bash
sudo apt update
rosdep update && rosdep install -i -r --from-paths src --rosdistro humble -y
colcon build --symlink-install --packages-up-to isaac_ros_visual_slam isaac_ros_nvblox
source install/setup.bash
```

---

## 6. cuVSLAM + nvblox with the D435i

The D435i is the right camera for this: it has a stereo IR pair plus an IMU,
which is exactly cuVSLAM's input. Inside the container:

```bash
sudo apt install -y ros-humble-realsense2-camera ros-humble-librealsense2*

# terminal 1
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py

# terminal 2
ros2 launch nvblox_examples_bringup realsense_example.launch.py
```

Enable the IR emitter only if you also want depth; cuVSLAM tracking prefers it
off, and the 2.1 launch files expose that as an argument. Verify with:

```bash
ros2 topic hz /visual_slam/tracking/odometry
ros2 topic hz /nvblox_node/mesh
```

---

## 7. Memory budget — 16 GB unified, shared CPU+GPU

| Component | Unified RAM |
|---|---|
| Ubuntu + Unitree services + sensor drivers | ~3-4 GB |
| cuVSLAM | ~1-1.5 GB |
| nvblox (room at 5 cm, TSDF+ESDF) | ~2-4 GB |
| VoxelNeXt | ~2-3 GB |
| `gemma4:e4b` q4 + KV | ~5 GB |
| Cosmos3-Edge (BF16 — its only supported precision) | ~8-9 GB |

cuVSLAM + nvblox + a small LLM fits, with headroom. Adding VoxelNeXt makes it
tight. Adding Cosmos3-Edge does not fit at any combination, and it cannot be
quantized down — that model needs a Thor payload or the lab server.

Compute is the second wall: Orin NX 16 GB is 100 TOPS, and cuVSLAM + nvblox
already claim most of it. Watch it with `tegrastats` before adding anything.

---

## 8. If the robot turns out to be on something else

- **L4T 35.3.1 (JP 5.1.1)** — this is what the robot actually runs. See §10 for
  the in-place upgrade to 35.4.1. The alternative is Isaac ROS `release-2.0`,
  pinned to a tag rather than a branch, with no system changes at all.
- **Already on JetPack 6.x.** Use `release-3.x`/`4.x` per the table; your
  existing v4.5-0 checkout may be reusable.
- **You need current Isaac ROS features.** Current releases require JetPack
  7.2, which means either a reflash (don't) or a Jetson Thor payload — which is
  what NVIDIA's own
  [G1 cloud-control tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_cloud_control_bringup/index.html)
  assumes, with the heavy stack off-robot.

---

## 9. Before you change anything on the robot

```bash
# record what is running and what is installed, so it can be compared later
systemctl list-unit-files --state=enabled > ~/before_services.txt
dpkg -l > ~/before_packages.txt
cat /etc/nv_tegra_release > ~/before_l4t.txt
```

Keep those on the laptop, not only on the robot. If Unitree's voice or loco
services stop after a change, the diff of `before_services.txt` is the fastest
way back.

Everything in §5 is additive — a directory in `$HOME` and Docker images. It is
undone by deleting `~/workspaces/isaac_ros-dev` and running `docker system
prune -a`. The only step in this document that changes the base system is the
optional apt bump in §8; treat that as the point of no easy return.

---

## 10. In-place L4T upgrade: 35.3.1 → 35.4.1 (JetPack 5.1.1 → 5.1.2)

This is the chosen path to reach Isaac ROS 2.1. It is an apt upgrade, not a
reflash — but it is still the riskiest step in this document, because
`nvidia-l4t-bootloader` is installed and its upgrade **writes the boot
partitions**.

### Read this before starting

- **The failure mode is a Jetson that will not boot.** If power is lost or the
  dpkg transaction is interrupted while the bootloader package is being
  configured, recovery requires force-recovery mode plus an SDK Manager flash
  from an x86 Ubuntu host — the exact reflash this document exists to avoid.
- **Put the robot on wall power** for the whole operation. Not battery.
- **Have the recovery path ready before you start:** physical access to the
  Jetson's recovery button and USB-C port, and an x86 Ubuntu machine with
  NVIDIA SDK Manager installed. If you do not have both, use Isaac ROS 2.0
  instead and change nothing on the system.
- **Run everything inside `tmux`.** The Jetson has two default routes (wlan0
  and eth0); an SSH drop mid-`dpkg` is how half-configured systems happen.
- **The kernel moves 5.10.104-tegra → 5.10.120-tegra**, and this robot has
  out-of-tree modules: `8852bu` (the RTL8852BU USB wifi driving `wlan0`, and
  the robot's only internet path) and `rtk_btusb`. Both are DKMS with sources
  in `/usr/src`, but **their DKMS rebuild fails**, because Unitree hardcoded
  the old kernel path into the driver Makefiles:

  ```
  make -C /lib/modules/5.10.104-tegra/build M=... modules
  make[1]: *** /lib/modules/5.10.104-tegra/build: No such file or directory.
  ```

  DKMS passes `KVERSION=5.10.120-tegra` and the Makefile ignores it. Workaround
  that gets wifi back immediately:

  ```bash
  sudo mkdir -p /lib/modules/5.10.104-tegra
  sudo ln -sfn /usr/src/linux-headers-5.10.120-tegra-ubuntu20.04_aarch64/kernel-5.10 \
               /lib/modules/5.10.104-tegra/build
  sudo dkms autoinstall -k 5.10.120-tegra
  ```

  The real fix is to make those Makefiles honour `$(KVERSION)`; otherwise this
  recurs on every kernel change. Note `dkms status` run as a normal user also
  misreports rtl8852bu as missing its `dkms.conf` — a permission artifact, the
  file is there.
- **`r8169` does not exist in 35.4.1** — L4T ships `r8168.ko` instead. Verified
  safe on this robot: the NIC is `[10ec:8168]` and `modules.alias` maps
  `pci:v000010ECd00008168` to `r8168`, so `eth0` still comes up after reboot.
- **Back up `/boot/Image` before starting.** The upgrade replaces it, and the
  old kernel's module tree is stripped down to just the two DKMS modules, so
  there is no rollback once `nvidia-l4t-kernel` is configured.
- **The SSH lifeline survives regardless.** `eth0` uses `r8169`, an in-tree
  driver, so the wired 192.168.123.164 link comes back even if wifi does not.
  Do this work over the wire, never over wifi.
- `librealsense` is installed from the RealSense apt repo here, so it can
  simply be reinstalled if it misbehaves.

### Verified facts this procedure relies on

- The robot has **no NVIDIA L4T apt source configured** — 40 `nvidia-l4t-*`
  packages are installed at 35.3.1 with nothing to upgrade from. The source
  must be added first.
- `r35.4` is published for both `common` and `t234` (Orin is `t234`, despite
  `/etc/nv_tegra_release` reporting the legacy `BOARD: t186ref` string).
- The OTA signing key `jetson-ota-public.asc` is already in
  `/etc/apt/trusted.gpg.d/`, so no key import is needed.
- `/` is a 1.9 TB NVMe with 1.8 TB free. Space is not a constraint.

### Procedure

```bash
# 0. from the laptop, over the WIRED link, in tmux
ssh unitree@192.168.123.164
tmux new -s l4t

# 1. snapshot, and copy these to the laptop afterwards
systemctl list-unit-files --state=enabled > ~/before_services.txt
dpkg -l > ~/before_packages.txt
cp /etc/nv_tegra_release ~/before_l4t.txt
sudo dkms status                      # confirm rtl8852bu + rtkbtusb are registered

# 1b. kernel rollback path -- extlinux.conf already ships a commented backup
#     entry, and TIMEOUT 30 gives you a serial-console menu to pick it
sudo cp /boot/Image /boot/Image.backup
sudo cp /boot/initrd /boot/initrd.backup
sudo cp /boot/extlinux/extlinux.conf ~/extlinux.conf.backup
# then uncomment the "LABEL backup" block in /boot/extlinux/extlinux.conf
# and point its INITRD at /boot/initrd.backup

# 2. add the L4T apt source (Orin = t234)
sudo tee /etc/apt/sources.list.d/nvidia-l4t-apt-source.list >/dev/null <<'EOF'
deb https://repo.download.nvidia.com/jetson/common r35.4 main
deb https://repo.download.nvidia.com/jetson/t234 r35.4 main
EOF
sudo apt update

# 3. DO NOT use dist-upgrade. Measured on this robot, it removes 20 packages
#    (the whole libopencv-*-dev set) and drags in an unrelated focal-updates
#    upgrade of libc6, udev, glib, bluez and gnupg. Upgrade only L4T:
PKGS=$(dpkg -l | awk '/^ii  nvidia-l4t/{print $2}' | tr '\n' ' ')
echo "$PKGS" | wc -w                       # expect 40
apt-get -s install --only-upgrade $PKGS > /tmp/l4t-targeted.txt
grep -c ^Remv /tmp/l4t-targeted.txt        # expect 0
grep ^Inst /tmp/l4t-targeted.txt | grep -v nvidia-l4t   # expect no output
```

Verified on this robot: the targeted form gives **0 removals, 40 installs, all
of them `nvidia-l4t-*`**. That is the command to run.

```bash
# 4. the irreversible part -- do not interrupt, do not power off
sudo apt-get install --only-upgrade $PKGS
sudo reboot
```

Afterwards, either remove `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list`
or pin it, so that a future absent-minded `apt upgrade` does not pull the rest
of r35.4 along with the OpenCV removals.

### Verify afterwards

```bash
cat /etc/nv_tegra_release          # want: R35 (release), REVISION: 4.1
dpkg -l | grep nvidia-l4t-core     # want: 35.4.1-*
ls -d /usr/local/cuda-*            # still 11.4
realsense-viewer --version 2>/dev/null || rs-enumerate-devices | head
```

Once `/etc/nv_tegra_release` reports REVISION 4.1, §5's `release-2.1` clone and
`run_dev.sh` will match the host, and the rest of this document applies as
written.

### Also worth doing while you are in there

Docker's default runtime is `runc`; the `nvidia` runtime is present but not
default. Isaac ROS's `run_dev.sh` passes the runtime explicitly, so this is
optional, but it removes a class of confusing GPU-missing errors:

```bash
# /etc/docker/daemon.json
{ "default-runtime": "nvidia" }
```
