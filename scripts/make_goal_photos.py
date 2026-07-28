"""Goal photo sheets — the human-intuitive check for goal placement.

For each scene in outputs/goal_maps/goals.json, produce one image:
  LEFT : the camera frame ~12 frames BEFORE the goal, with the chosen goal
         projected into it as a marker (circle + label) — "the goal, as the
         robot will see it while approaching"
  RIGHT: the camera frame AT the goal moment — "standing at the goal"

Judge each in seconds: is the marker on ground you'd send a robot to?

Needs locally: data/clips/<scene>.mp4, data/poses/<scene>_poses.npz.
Usage (Mac): python scripts/make_goal_photos.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CAMERA_HEIGHT_M = 0.25
LOOKBACK = 12


def sample_frames(video_path: Path, num=81):
    import imageio.v3 as iio
    frames = list(iio.imiter(str(video_path)))
    idxs = np.linspace(0, len(frames) - 1, num, dtype=int)
    return [frames[i] for i in idxs]


def center_crop(img, tw=560, th=336):
    from PIL import Image
    im = Image.fromarray(img)
    w, h = im.size
    s = max(tw / w, th / h)
    im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    l, t = (im.width - tw) // 2, (im.height - th) // 2
    return np.array(im.crop((l, t, l + tw, t + th)))


def main():
    from PIL import Image, ImageDraw

    goals = json.load(open(REPO_ROOT / "outputs/goal_maps/goals.json"))
    out_dir = REPO_ROOT / "outputs/goal_maps"
    done = skipped = 0
    for scene, g in goals.items():
        clip = REPO_ROOT / f"data/clips/{scene}.mp4"
        poses = REPO_ROOT / f"data/poses/{scene}_poses.npz"
        if not clip.exists() or not poses.exists():
            skipped += 1
            continue
        try:
            frames = sample_frames(clip)
        except Exception as e:
            print(f"  SKIP {scene}: unreadable clip ({type(e).__name__})")
            skipped += 1
            continue
        d = np.load(poses)
        K, w2c = d["K"], d["w2c"]
        cam_z = d["cam_positions"][:, 2]
        j = g["goal_frame"]
        k = max(0, j - LOOKBACK)
        left = center_crop(frames[k]); right = center_crop(frames[min(j, len(frames) - 1)])

        # project the goal point (at true local ground height) into frame k
        gz = float(cam_z[j] - CAMERA_HEIGHT_M)
        p = np.array([g["goal_xy"][0], g["goal_xy"][1], gz, 1.0])
        img = Image.fromarray(left)
        draw = ImageDraw.Draw(img, "RGBA")
        # Ground ring: a REAL 0.3 m-radius circle lying ON the ground at the goal,
        # each rim point projected separately -> a foreshortened ellipse that sits
        # in the scene like a hula hoop on the grass. This is the scale cue: it
        # should read as a ~60 cm-wide hoop a Go2 could stand inside.
        ring = []
        behind = False
        for ang in np.linspace(0, 2 * np.pi, 32):
            q = np.array([g["goal_xy"][0] + 0.3 * np.cos(ang),
                          g["goal_xy"][1] + 0.3 * np.sin(ang), gz, 1.0])
            qc = (w2c[k] @ q)[:3]
            if qc[2] <= 0.05:
                behind = True
                break
            ring.append((K[0, 0] * qc[0] / qc[2] + K[0, 2],
                         K[1, 1] * qc[1] / qc[2] + K[1, 2]))
        if not behind and ring:
            draw.polygon(ring, outline=(0, 255, 0, 255))
            for i in range(len(ring)):
                draw.line([ring[i], ring[(i + 1) % len(ring)]],
                          fill=(0, 255, 0, 255), width=3)
        draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0, 190))
        draw.text((4, 2), f"{scene}  goal frame {j} ({g['dist_m']} m, walk {g['walkability']:.0%})"
                          f"  LEFT: approach view +marker  RIGHT: at the goal",
                  fill=(255, 255, 255, 255))

        sheet = Image.new("RGB", (560 * 2 + 4, 336), (0, 0, 0))
        sheet.paste(img, (0, 0)); sheet.paste(Image.fromarray(right), (564, 0))
        sheet.save(out_dir / f"{scene}_goalphoto.png")
        done += 1
    print(f"wrote {done} goal photo sheets to {out_dir} ({skipped} skipped, missing clip/poses)")


if __name__ == "__main__":
    main()
