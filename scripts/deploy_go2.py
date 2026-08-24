"""Deploy a trained policy on the Go2W: camera + odometry in, velocity out.

Publishes geometry_msgs/Twist on /cmd_vel — the input of soar-go2's
motion_control.py (go2w_sdk), which owns the Unitree SDK and enforces a
deadman timeout + /estop. Run motion_control first; this node replaces the
teleop source. (/joystick_cmd_vel is only joy_bridge's RECORDING stream —
nothing drives the robot from it.) No diffusion runs on the robot — only the
small PPO network; the measured per-inference milliseconds are logged.

The policy decides at --rate Hz, but the last command is REPUBLISHED at 20 Hz
so motion_control's deadman never trips between decisions.

Loop at --rate Hz:
  1. latest camera frame  -> center-crop 5:3 -> resize 560x336  (same
     preprocessing as prepare_rosbag_clips.py, so the policy sees the
     training distribution)
  2. goal vector: the goal is fixed in the odom frame at startup
     (--goal_dx/--goal_dy meters, robot frame at t=0); each tick it is
     re-expressed in the CURRENT robot frame from /Odometry -> (dx, dy, bearing)
  3. action = policy(rgb, goal); v = a0 * 0.25 * rate, w = a1 * 0.3 * rate
     (the sim step is 0.25 m / 0.3 rad; scaling by rate converts steps/s to
     m/s and rad/s), clamped to --max_v / --max_w
  4. publish Twist on /cmd_vel (kept alive at 20 Hz for the deadman)

Safety: --dry_run computes but never publishes. /estop (std_msgs/Bool) is
motion_control's kill switch — keep a terminal ready with:
  ros2 topic pub /estop std_msgs/msg/Bool "data: true" --once
Zero command is sent when the goal is within 0.75 m and on Ctrl-C.

Usage on the Jetson:
    python3 scripts/deploy_go2.py \
        --checkpoint ppo_800000_steps.zip --goal_dx 5.0 --goal_dy 0.0 \
        [--rate 2.0] [--dry_run]
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def quat_to_yaw(x, y, z, w) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def preprocess(bgr: np.ndarray, W: int = 560, H: int = 336) -> np.ndarray:
    import cv2
    h, w = bgr.shape[:2]
    ar = W / H
    ch = int(w / ar)
    if ch <= h:
        y0 = (h - ch) // 2
        crop = bgr[y0:y0 + ch]
    else:
        cw = int(h * ar)
        x0 = (w - cw) // 2
        crop = bgr[:, x0:x0 + cw]
    rgb = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return np.ascontiguousarray(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="policy .zip (required unless --baseline)")
    ap.add_argument("--baseline", action="store_true",
                    help="IGNORE the camera: pure goal-bearing servo "
                         "(v ~ distance, w ~ bearing). The control lap for "
                         "the outdoor comparison — any difference between this "
                         "trajectory and the policy's is what world-model "
                         "training contributed.")
    ap.add_argument("--goal_dx", type=float, required=True,
                    help="goal x (m) in the robot frame at startup (forward)")
    ap.add_argument("--goal_dy", type=float, required=True,
                    help="goal y (m) in the robot frame at startup (left)")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--max_v", type=float, default=0.6)
    ap.add_argument("--max_w", type=float, default=0.8)
    ap.add_argument("--goal_radius", type=float, default=0.75)
    ap.add_argument("--image_topic", default="/camera/camera/color/image_raw/compressed")
    ap.add_argument("--odom_topic", default="/Odometry")
    ap.add_argument("--cmd_topic", default="/cmd_vel")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="command blending 0..1: published = smooth*new + "
                         "(1-smooth)*previous. 1.0 = raw policy output; lower "
                         "= gentler transitions (soar-go2's RL rate-limits "
                         "velocity changes the same way in training)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from stable_baselines3 import PPO

    if args.baseline:
        model = None
        print("[deploy] BASELINE mode: goal-bearing servo, camera ignored")
    else:
        assert args.checkpoint, "--checkpoint required unless --baseline"
        model = PPO.load(args.checkpoint, device="cuda")
        print(f"[deploy] loaded {args.checkpoint}")

    class PolicyNode(Node):
        def __init__(self):
            super().__init__("nav_policy")
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(CompressedImage, args.image_topic, self.on_img, qos)
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
            self.pub = self.create_publisher(Twist, args.cmd_topic, 10)
            self.img = None
            self.pose = None            # (x, y, yaw) in odom frame
            self.goal_odom = None       # set on first odom using startup frame
            self.lat = []
            self.cmd = (0.0, 0.0)       # latest decided (v, w)
            self.create_timer(1.0 / args.rate, self.tick)
            # deadman keep-alive: motion_control stops if /cmd_vel goes quiet,
            # so re-send the current command at 20 Hz between policy decisions
            self.create_timer(0.05, self.keepalive)

        def on_img(self, msg):
            self.img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

        def on_odom(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            self.pose = (p.x, p.y, yaw)
            if self.goal_odom is None:
                c, s = np.cos(yaw), np.sin(yaw)
                self.goal_odom = (p.x + c * args.goal_dx - s * args.goal_dy,
                                  p.y + s * args.goal_dx + c * args.goal_dy)
                self.get_logger().info(f"goal fixed at odom ({self.goal_odom[0]:.2f}, "
                                       f"{self.goal_odom[1]:.2f})")

        def tick(self):
            if self.img is None or self.pose is None or self.goal_odom is None:
                self.get_logger().info("waiting for camera/odom ...", throttle_duration_sec=2.0)
                return
            x, y, yaw = self.pose
            dxw, dyw = self.goal_odom[0] - x, self.goal_odom[1] - y
            dist = float(np.hypot(dxw, dyw))
            c, s = np.cos(-yaw), np.sin(-yaw)
            dx, dy = c * dxw - s * dyw, s * dxw + c * dyw   # goal in robot frame
            bearing = float(np.arctan2(dy, dx))
            if dist < args.goal_radius:
                self.publish(0.0, 0.0)
                self.get_logger().info(f"GOAL REACHED ({dist:.2f} m) — holding")
                return
            t0 = time.perf_counter()
            if args.baseline:
                v = float(np.clip(0.5 * dist, 0.0, args.max_v))
                w = float(np.clip(1.5 * bearing, -args.max_w, args.max_w))
            else:
                obs = {"rgb": preprocess(self.img),
                       "goal": np.array([dx, dy, bearing], dtype=np.float32)}
                action, _ = model.predict(obs, deterministic=True)
                v = float(np.clip(action[0] * 0.25 * args.rate, -args.max_v, args.max_v))
                w = float(np.clip(action[1] * 0.30 * args.rate, -args.max_w, args.max_w))
            ms = (time.perf_counter() - t0) * 1e3
            self.lat.append(ms)
            a = float(np.clip(args.smooth, 0.0, 1.0))
            v = a * v + (1.0 - a) * self.cmd[0]
            w = a * w + (1.0 - a) * self.cmd[1]
            self.publish(v, w)
            self.get_logger().info(
                f"dist {dist:4.1f} m bearing {np.degrees(bearing):+5.0f} deg | "
                f"v {v:+.2f} w {w:+.2f} | policy {ms:.1f} ms "
                f"(median {np.median(self.lat):.1f})")

        def publish(self, v, w):
            self.cmd = (v, w)
            self.keepalive()

        def keepalive(self):
            if args.dry_run:
                return
            msg = Twist()
            msg.linear.x = self.cmd[0]
            msg.angular.z = self.cmd[1]
            self.pub.publish(msg)

    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass    # Ctrl-C can tear down the ROS context before this runs
        if node.lat:
            print(f"[deploy] policy latency: median {np.median(node.lat):.1f} ms "
                  f"over {len(node.lat)} inferences")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
