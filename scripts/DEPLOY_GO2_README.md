# Running the nav policy on the Go2W

Policy = PPO trained inside the diffusion world model. On the robot it is only
a small network (~8 ms/inference on Thor) — no diffusion runs at deploy time.

Files live on Thor in `~/nav_policy/`:
- `deploy_go2.py` — the policy node (camera + /Odometry in → Twist on /cmd_vel)
- `checkpoints/` — policy .zip files (e.g. `ppo_800000_steps.zip` = the
  trail_00 cached champion, first deployed 2026-08-23)
- `README.md` — this file

## One-time python setup (already done 2026-08-23)

The checkpoints are saved with numpy 2; the system python must keep numpy 1
(tracker needs it). A venv shadows numpy only:

```bash
python3 -m venv --system-site-packages ~/nav_env
~/nav_env/bin/pip install "numpy>=2"
~/nav_env/bin/pip install --no-deps stable-baselines3
~/nav_env/bin/pip install gymnasium cloudpickle pandas
```

## Every session

`~/.bashrc` puts the conda site-packages on PYTHONPATH, which overrides the
venv's numpy — strip it in ANY terminal that runs the policy:

```bash
export PYTHONPATH=$(echo "$PYTHONPATH" | sed 's#:/home/soar/miniconda3/lib/python3.12/site-packages##')
```

## Bringup order

1. Robot on, **terrain mode** (controller).
2. **Orin** (`ssh unitree@192.168.123.18`, pw 123), three terminals:
   ```bash
   cd ~/soar-go2/ws/src/go2w_sdk/ && python3 go2w_sdk/motion_control.py
   ros2 launch livox_ros_driver2 msg_MID360_launch.py
   ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
   ```
3. **Thor** — camera:
   ```bash
   ros2 launch realsense2_camera rs_launch.py camera_name:='camera' rgb_camera.color_profile:=640x480x30
   ```
4. **Thor** — sanity (both must tick):
   ```bash
   ros2 topic hz /camera/camera/color/image_raw/compressed
   ros2 topic hz /Odometry
   ```
5. **Thor** — estop terminal, pre-typed:
   ```bash
   ros2 topic pub /estop std_msgs/msg/Bool "data: true" --once
   ```

## Run

Goal = (`--goal_dx` m forward, `--goal_dy` m left) of the robot's pose WHEN THE
NODE STARTS; it is then frozen in the odom frame (printed at startup). Robot
stops within 0.75 m of it.

```bash
cd ~/nav_policy
# dry run first: computes and prints, publishes nothing
~/nav_env/bin/python3 deploy_go2.py --checkpoint checkpoints/ppo_800000_steps.zip \
    --goal_dx 3.0 --goal_dy 0.0 --dry_run
# real run (gentle speed)
~/nav_env/bin/python3 deploy_go2.py --checkpoint checkpoints/ppo_800000_steps.zip \
    --goal_dx 3.0 --goal_dy 0.0 --max_v 0.4
```

Knobs: `--smooth 0.5` (default) blends commands between decisions — 1.0 = raw
policy output; `--max_v/--max_w` caps; `--rate` decision Hz (default 2).
Ctrl-C sends a zero command; motion_control's deadman also stops the robot if
the node dies.

Record a run:
```bash
ros2 bag record /camera/camera/color/image_raw/compressed /Odometry /cmd_vel -o nav_policy_test
```

## Troubleshooting

- `ModuleNotFoundError: numpy._core...` → the PYTHONPATH export above wasn't
  run in this terminal (conda numpy 1 is shadowing the venv).
- No camera topic → step 3; no /Odometry → FAST-LIO needs the LiDAR up first.
- Robot ignores commands → motion_control not running, or not in terrain mode.
- First inference ~300-500 ms is CUDA warmup; median settles ~8 ms.
