#!/usr/bin/env python3
"""
G1 Motion & Arm Control Skill CLI:
Direct CLI utility for moving the G1 humanoid, rotating, and executing arm gestures
via Unitree SDK2 LocoClient and G1ArmActionClient.

No ROS 2 runtime required. Run directly:

    python3 scripts/g1_motion_skill.py [interface] <command> [args...]

Usage Examples:
    # 1. Locomotion Move (dx forward in meters, dy left in meters):
    python3 scripts/g1_motion_skill.py enp2s0 move 0.15 0.0
    python3 scripts/g1_motion_skill.py enp2s0 move 0.20 0.0

    # 2. In-Place Rotation (degrees, positive = counter-clockwise, negative = clockwise):
    python3 scripts/g1_motion_skill.py enp2s0 rotate -30.0
    python3 scripts/g1_motion_skill.py enp2s0 rotate 45.0

    # 3. Arm Gestures:
    python3 scripts/g1_motion_skill.py enp2s0 shake_hand
    python3 scripts/g1_motion_skill.py enp2s0 wave low
    python3 scripts/g1_motion_skill.py enp2s0 wave high
    python3 scripts/g1_motion_skill.py enp2s0 release_arm

    # 4. State Checks & Transitions:
    python3 scripts/g1_motion_skill.py enp2s0 fsm
    python3 scripts/g1_motion_skill.py enp2s0 stand
    python3 scripts/g1_motion_skill.py enp2s0 damp
"""

import math
import os
import sys
import time

# Ensure unitree_sdk2_python is accessible
SDK_PATH = "/home/thakk100/Projects/unitree_sdk2_python"
if os.path.isdir(SDK_PATH) and SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

FSM_ZERO_TORQUE = 0
FSM_DAMP = 1
FSM_READY_STAND = 4
FSM_WALK = 501

DEFAULT_LINEAR_SPEED = 0.50      # m/s (Standard G1 continuous walking pace)
DEFAULT_YAW_RATE = 0.50          # rad/s
ROTATE_COMPENSATION = 5.0 / 3.0  # Ground-friction compensation factor
SETTLE_TIME = 2.0                # seconds between state transitions


class G1Motion:
    def __init__(self, network_interface: str = "enp2s0", domain_id: int = 0, timeout: float = 8.0):
        print(f"[*] Initializing DDS channel on interface '{network_interface}' (Domain {domain_id})...")
        ChannelFactoryInitialize(domain_id, network_interface)
        
        self.client = LocoClient()
        self.client.SetTimeout(timeout)
        self.client.Init()

        ret = self.client.SwitchToUserCtrl()
        if ret != 0:
            print(f"[!] Warning: SwitchToUserCtrl returned code {ret}")

        # Initialize Arm Client
        try:
            self.arm_client = G1ArmActionClient()
            self.arm_client.SetTimeout(timeout)
            self.arm_client.Init()
        except Exception as e:
            print(f"[!] Warning: Arm client initialization failed: {e}")
            self.arm_client = None

    def get_fsm(self) -> int:
        code, fsm_id = self.client.GetFsmId()
        if code != 0:
            raise RuntimeError(f"GetFsmId failed, code={code}")
        return fsm_id

    def ensure_walk_mode(self, max_attempts: int = 3) -> int:
        fsm_id = self.get_fsm()
        if fsm_id == FSM_WALK:
            return fsm_id

        if fsm_id != FSM_READY_STAND:
            print(f"[*] Transitioning FSM {fsm_id} -> ReadyStand (4)...")
            self.client.SetFsmId(FSM_READY_STAND)
            time.sleep(SETTLE_TIME)

        print("[*] Transitioning -> Walk Mode (501)...")
        for attempt in range(1, max_attempts + 1):
            self.client.SetFsmId(FSM_WALK)
            time.sleep(SETTLE_TIME)
            fsm_id = self.get_fsm()
            if fsm_id == FSM_WALK:
                print("[✓] Successfully in Walk Mode (501).")
                return fsm_id
            print(f"    Attempt {attempt}/{max_attempts}: FSM is {fsm_id}, retrying...")

        raise RuntimeError(f"Failed to enter walk mode. Current FSM: {fsm_id}")

    def move(self, dx: float, dy: float = 0.0, speed: float = 0.50):
        self.ensure_walk_mode()
        distance = math.hypot(dx, dy)
        if distance == 0:
            return

        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}")

        duration = distance / speed
        vx = (dx / distance) * speed
        vy = (dy / distance) * speed

        print(f"[*] Walking dx={dx:+.2f}m, dy={dy:+.2f}m (dist={distance:.2f}m) at user speed={speed:.2f}m/s for duration={duration:.2f}s...")
        # 20 Hz velocity streaming
        start_time = time.time()
        while time.time() - start_time < duration:
            self.client.SetVelocity(vx, vy, 0.0, duration=0.2)
            time.sleep(0.05)

        # Active graceful zero-velocity brake: Stream 0.0 m/s for 0.6s to firmly plant feet
        brake_start = time.time()
        while time.time() - brake_start < 0.6:
            self.client.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
            time.sleep(0.05)

        self.client.StopMove()
        time.sleep(0.2)
        print("[✓] Move completed and fully halted.")

    def rotate(self, degrees: float, yaw_rate: float = DEFAULT_YAW_RATE):
        self.ensure_walk_mode()
        target_rad = math.radians(degrees) * ROTATE_COMPENSATION
        if target_rad == 0:
            return

        rate = math.copysign(yaw_rate, target_rad)
        duration = abs(target_rad) / yaw_rate

        print(f"[*] Rotating {degrees:+.1f}° (compensated: {math.degrees(target_rad):+.1f}°) at {rate:.2f}rad/s (duration={duration:.2f}s)...")
        # 20 Hz streaming
        start_time = time.time()
        while time.time() - start_time < duration:
            self.client.SetVelocity(0.0, 0.0, rate, duration=0.2)
            time.sleep(0.05)

        # Active brake
        brake_start = time.time()
        while time.time() - brake_start < 0.5:
            self.client.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
            time.sleep(0.05)

        self.client.StopMove()
        time.sleep(0.2)
        print("[✓] Rotation completed and fully halted.")

    def shake_hand(self, hold_seconds: float = 3.5):
        print(f"[*] 🤝 Executing Handshake (hold={hold_seconds}s)...")
        if self.arm_client is not None:
            self.arm_client.ExecuteAction(action_map.get("shake hand", 27))
            time.sleep(hold_seconds)
            self.arm_client.ExecuteAction(action_map.get("release arm", 99))
        else:
            self.client.ShakeHand()
            time.sleep(hold_seconds)
            self.client.ShakeHand()
        print("[✓] Handshake completed and arm released.")

    def wave(self, wave_type: str = "low", hold_seconds: float = 3.0):
        print(f"[*] 👋 Executing Wave ({wave_type}, hold={hold_seconds}s)...")
        if self.arm_client is not None:
            if wave_type in ("high", "high_wave"):
                self.arm_client.ExecuteAction(action_map.get("high wave", 26))
            else:
                self.arm_client.ExecuteAction(action_map.get("face wave", 25))
            time.sleep(hold_seconds)
            self.arm_client.ExecuteAction(action_map.get("release arm", 99))
        else:
            self.client.WaveHand(False)
            time.sleep(hold_seconds)
        print("[✓] Wave completed and arm released.")

    def release_arm(self):
        print("[*] Releasing arm to neutral...")
        if self.arm_client is not None:
            self.arm_client.ExecuteAction(action_map.get("release arm", 99))
        print("[✓] Arm released.")

    def stand(self):
        print("[*] Setting FSM to ReadyStand (4)...")
        self.client.SetFsmId(FSM_READY_STAND)
        time.sleep(SETTLE_TIME)
        print(f"[✓] FSM is now {self.get_fsm()}")

    def damp(self):
        print("[*] Setting FSM to Damp (1)...")
        self.client.Damp()
        time.sleep(1.0)
        print(f"[✓] FSM is now {self.get_fsm()}")


def print_help():
    print("""
G1 Motion & Arm Control CLI
============================
Usage:
    python3 scripts/g1_motion_skill.py [interface] <command> [args...]

Commands:
    move <dx> [dy] [speed]   - Walk relative dx meters (forward/backward) and dy meters (left/right)
    rotate <degrees> [rate]  - Rotate in place by degrees (+ = CCW/left, - = CW/right)
    shake_hand [hold_sec]    - Perform handshake greeting gesture
    wave [low|high]          - Perform waving gesture (low/face wave or high wave)
    release_arm              - Release arms to neutral resting position
    fsm                      - Query current robot state machine ID
    stand                    - Transition robot to ReadyStand mode (FSM 4)
    damp                     - Transition robot to Damp mode (FSM 1)

Examples:
    python3 scripts/g1_motion_skill.py enp2s0 move 0.15 0.0
    python3 scripts/g1_motion_skill.py enp2s0 rotate -30.0
    python3 scripts/g1_motion_skill.py enp2s0 shake_hand
    python3 scripts/g1_motion_skill.py enp2s0 wave low
    """)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    # Determine network interface (default 'enp2s0')
    iface = "enp2s0"
    if args[0].startswith("enp") or args[0].startswith("eth") or args[0].startswith("wlo") or args[0] == "lo":
        iface = args.pop(0)

    if not args:
        print_help()
        sys.exit(0)

    cmd = args[0].lower()
    robot = G1Motion(network_interface=iface)

    if cmd == "move":
        dx = float(args[1]) if len(args) > 1 else 0.20
        dy = float(args[2]) if len(args) > 2 else 0.0
        speed = float(args[3]) if len(args) > 3 else DEFAULT_LINEAR_SPEED
        robot.move(dx, dy, speed)

    elif cmd in ("rotate", "turn"):
        deg = float(args[1]) if len(args) > 1 else 30.0
        rate = float(args[2]) if len(args) > 2 else DEFAULT_YAW_RATE
        robot.rotate(deg, rate)

    elif cmd in ("shake_hand", "shake", "handshake"):
        hold = float(args[1]) if len(args) > 1 else 3.5
        robot.shake_hand(hold)

    elif cmd in ("wave", "low_wave", "high_wave"):
        wtype = args[1] if len(args) > 1 else ("high" if cmd == "high_wave" else "low")
        robot.wave(wtype)

    elif cmd in ("release", "release_arm"):
        robot.release_arm()

    elif cmd in ("fsm", "status"):
        print(f"[*] Current Robot FSM ID: {robot.get_fsm()} (0=ZeroTorque, 1=Damp, 4=ReadyStand, 501=Walk)")

    elif cmd in ("stand", "ready"):
        robot.stand()

    elif cmd == "damp":
        robot.damp()

    else:
        print(f"[!] Unknown command '{cmd}'")
        print_help()
        sys.exit(-1)


if __name__ == "__main__":
    main()
