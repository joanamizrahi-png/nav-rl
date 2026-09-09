"""Confirm a camera topic is publishing, and that we decode its colours right.

Run this on the robot BEFORE deploying a policy, especially after a camera
change. It answers three questions in one go:

  1. is anything actually publishing on that topic (the node otherwise sits on
     "waiting for camera/odom" forever, with no error);
  2. what colour order the stream is in -- deploy_go2.py assumes cv2.imdecode
     returns BGR and flips to RGB, so an already-RGB stream would feed the
     policy swapped channels and simply behave oddly;
  3. what resolution and aspect it is, since preprocess center-crops to 5:3.

    python3 scripts/check_camera_topic.py --topic /camera/camera/color/image_raw/compressed

Writes /tmp/frame_check.png, interpreted exactly the way the deploy node reads
it: if grass is green and sky is blue, the decode is right.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/odin1/image/undistorted")
    ap.add_argument("--out", default="/tmp/frame_check.png")
    ap.add_argument("--policy_out", default="/tmp/frame_policy_view.png",
                    help="also save the cropped+resized frame the policy actually receives")
    ap.add_argument("--obs_w", type=int, default=336)
    ap.add_argument("--obs_h", type=int, default=224)
    ap.add_argument("--timeout_s", type=float, default=10.0)
    args = ap.parse_args()

    import cv2
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image

    compressed = args.topic.endswith("/compressed")
    msg_type = CompressedImage if compressed else Image
    print(f"subscribing to {args.topic} as {msg_type.__name__}")

    rclpy.init()
    node = Node("check_camera_topic")
    got = {}

    def on_msg(msg):
        if compressed:
            got["fmt"] = msg.format
            got["img"] = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            got["fmt"] = msg.encoding
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            got["img"] = arr[:, :, :3]

    node.create_subscription(msg_type, args.topic, on_msg,
                             QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    steps = int(args.timeout_s / 0.05)
    for _ in range(steps):
        rclpy.spin_once(node, timeout_sec=0.05)
        if "img" in got:
            break

    if "img" not in got:
        print(f"NO FRAME in {args.timeout_s:.0f} s.")
        print("  - check `ros2 topic list` for the exact name")
        print("  - check `ros2 topic info <topic>` for the message type; this script")
        print("    picks CompressedImage only when the name ends in /compressed")
    else:
        img = got["img"]
        h, w = img.shape[:2]
        print(f"format/encoding: {got['fmt']}")
        print(f"resolution: {w}x{h}  (aspect {w / h:.2f}; the policy crops to 1.67)")
        b, g, r = (float(img[:, :, i].mean()) for i in range(3))
        print(f"channel means: ch0 {b:.0f}  ch1 {g:.0f}  ch2 {r:.0f}")
        cv2.imwrite(args.out, img)
        # Exactly what deploy_go2.py feeds the network: center-crop to the
        # policy's aspect, then resize. Saved back in BGR so it looks normal.
        ar = args.obs_w / args.obs_h
        ch = int(w / ar)
        if ch <= h:
            y0 = (h - ch) // 2
            crop = img[y0:y0 + ch]
        else:
            cw = int(h * ar)
            x0 = (w - cw) // 2
            crop = img[:, x0:x0 + cw]
        small = cv2.resize(crop, (args.obs_w, args.obs_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(args.policy_out, small)
        big = cv2.resize(small, (args.obs_w * 2, args.obs_h * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(args.policy_out.replace(".png", "_2x.png"), big)
        print(f"crop kept {crop.shape[1]}x{crop.shape[0]} of {w}x{h} "
              f"({100 * crop.shape[0] / h:.0f}% of height, {100 * crop.shape[1] / w:.0f}% of width)")
        print(f"wrote {args.policy_out} ({args.obs_w}x{args.obs_h}, what the policy sees)")
        print(f"wrote {args.policy_out.replace('.png', '_2x.png')} (same, doubled for viewing)")
        print(f"wrote {args.out} -- open it. Grass green and sky blue means the")
        print("decode matches what deploy_go2.py assumes. An orange sky means the")
        print("channels are swapped: run deploy_go2.py with --swap_rb.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
