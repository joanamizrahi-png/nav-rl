# Deploying the nav policy on the Go2W — runbook

The policy is a small network trained inside the diffusion world model. On the
robot it is only that network (~8 ms per inference on Thor); **no diffusion runs
at deploy time**. It takes a camera frame and a relative goal, and outputs a
forward and a yaw command on `/cmd_vel`.

Follow the sections in order. Do not skip section 3.

Files on Thor live in `~/nav_policy/`:

```
~/nav_policy/
  deploy_go2.py                        the policy node
  <run_name>/
    checkpoints/ppo_XXXXXX_steps.zip   the policy
    env_config.json                    how it was trained (read automatically)
```

`env_config.json` must sit one directory above `checkpoints/`, because that is
where the node looks for it to recover the step size and yaw step.

---

## 1. Before you leave the lab

**Copy the current code.** `deploy_go2.py` changes; an old copy on Thor will
silently use the wrong image size.

```bash
scp scripts/deploy_go2.py soar@<thor>:~/nav_policy/
```

**Copy the policy and its config together**, from wherever the run lives:

```bash
scp "marlowe:/scratch/m000204-pm06b/joana/outputs/<run>/checkpoints/ppo_XXXXXX_steps.zip" .
scp "marlowe:/scratch/m000204-pm06b/joana/outputs/<run>/env_config.json" .
scp -r <run> soar@<thor>:~/nav_policy/
```

**Write down what you are testing and what would count as success** before the
robot is switched on: the goal distance, the route you expect, and the failure
you are watching for. A run nobody predicted is hard to learn from.

---

## 2. Bringup

1. Robot on, **terrain mode** on the controller.
2. **Orin** (`ssh unitree@192.168.123.18`, pw `123`), three terminals:
   ```bash
   cd ~/soar-go2/ws/src/go2w_sdk/ && python3 go2w_sdk/motion_control.py
   ros2 launch livox_ros_driver2 msg_MID360_launch.py
   ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
   ```
3. **Thor** — camera:
   ```bash
   ros2 launch realsense2_camera rs_launch.py camera_name:='camera' \
       rgb_camera.color_profile:=640x480x30
   ```
4. **Thor** — an estop terminal, command pre-typed and not yet run:
   ```bash
   ros2 topic pub /estop std_msgs/msg/Bool "data: true" --once
   ```

**Every terminal that runs the policy** needs the conda path stripped, or
conda's numpy 1 shadows the venv's numpy 2 and the checkpoint will not load:

```bash
export PYTHONPATH=$(echo "$PYTHONPATH" | sed 's#:/home/soar/miniconda3/lib/python3.12/site-packages##')
```

The `~/nav_env` venv exists for that reason (created 2026-08-23): system
site-packages plus numpy 2, `stable-baselines3` installed with `--no-deps`,
plus gymnasium, cloudpickle and pandas. Run the policy with
`~/nav_env/bin/python3`, never the system python. To rebuild it:

```bash
python3 -m venv --system-site-packages ~/nav_env
~/nav_env/bin/pip install "numpy>=2"
~/nav_env/bin/pip install --no-deps stable-baselines3
~/nav_env/bin/pip install gymnasium cloudpickle pandas
```

---

## 3. Pre-flight, every session

**Both topics must tick.** No output means no data, and the node waits forever:

```bash
ros2 topic hz /camera/camera/color/image_raw/compressed
ros2 topic hz /Odometry
```

**Dry run.** Computes and prints commands, publishes nothing, robot cannot
move:

```bash
cd ~/nav_policy
~/nav_env/bin/python3 deploy_go2.py --checkpoint <run>/checkpoints/ppo_XXXXXX_steps.zip \
    --goal_dx 3.0 --goal_dy 0.0 --dry_run
```

**Read these three lines before continuing:**

```
[deploy] observation from checkpoint: 336x224 (WxH)
[deploy] step 0.25 m, yaw 0.3 rad per decision  [.../env_config.json]
[deploy] at 2.0 Hz -> max v 0.50 m/s, max w 0.60 rad/s
```

- The resolution comes from the checkpoint itself. 560x336 is the *render* size
  and 336x224 is the *observation* size; an older copy of this node hardcoded
  the former, which fed recent policies a differently framed picture than they
  trained on.
- If the second line says `[CLI default]`, `env_config.json` was not found and
  the action scaling is a guess. Go back and copy it.
- The third line is the speed you are about to allow. Decide it is acceptable
  for the space you are standing in.

Also check the goal line the node prints. The goal is (`--goal_dx` forward,
`--goal_dy` left) of the robot's pose **at node start**, then frozen in the odom
frame. If the robot is not pointing where you think, the goal is not where you
think.

---

## 4. Run

**Always run the baseline first.** It ignores the camera and drives to the goal
on odometry alone. It verifies odometry, the goal frame, the deadman keep-alive
and your safe distances, with no network in the loop. It is also the control
your results need: if the baseline reaches the goal too, the task did not
require vision.

```bash
~/nav_env/bin/python3 deploy_go2.py --baseline \
    --goal_dx 4.0 --goal_dy 0.0 --max_v 0.4 --log baseline_1.csv
```

**Then the policy, on the identical goal:**

```bash
~/nav_env/bin/python3 deploy_go2.py --checkpoint <run>/checkpoints/ppo_XXXXXX_steps.zip \
    --goal_dx 4.0 --goal_dy 0.0 \
    --rate 2 --max_v 0.5 --max_w 0.6 --smooth 0.5 \
    --timeout_s 60 --no_progress_s 15 --log policy_1.csv
```

Record video and topics for anything you may want to show or analyse:

```bash
ros2 bag record /camera/camera/color/image_raw/compressed /Odometry /cmd_vel -o nav_policy_test
```

Start easy and add difficulty one step at a time: open pavement straight ahead,
then a goal with grass on one side, then a goal past a bend. Change one thing
per run.

---

## 5. Stopping

- **Ctrl-C** in the policy terminal sends a zero command and exits.
- **Estop** topic in the pre-typed terminal.
- **`--timeout_s`** stops the robot after that many seconds (default 60).
- **`--no_progress_s`** stops it if the goal distance has not improved by 0.25 m
  within that window (default 15). This is the circling guard: in simulation the
  policy looped and overshot on corner goals, and circling is hard to call by
  eye in the first few seconds.
- If the node dies, motion_control's own deadman stops the robot because
  `/cmd_vel` goes quiet.

---

## 6. After each run

Keep the `--log` CSV and the bag, and write one line about what happened while
it is fresh. The CSV has `t, x, y, yaw, dist, bearing, v, w, ms` for every
decision, which is enough to tell circling from oscillation from driving into
something, without relying on memory.

---

## 7. Choosing the rate

The policy was trained to move a fixed amount per decision: `step_size_m`
forward and `yaw_step_rad` of turn at full command. The node converts that to a
velocity as `v = action * step_size_m * rate`, so **`--rate` is a speed knob**.
At 2 Hz with a 0.25 m step, full forward is 0.5 m/s.

Distance travelled per decision is `action * step_size_m` whatever rate you
choose, so a slower rate does **not** give the policy more decisions per metre.
What it gives is more time for the robot to actually reach the commanded
velocity, and fresher camera and odometry at each decision.

So choose it this way:

- Start at 2 Hz.
- Afterwards check the log: the median distance between consecutive `(x, y)`
  should be roughly `step_size_m` scaled by the typical `|v| / max_v`. If the
  robot consistently moves less than commanded, it is not tracking the velocity;
  lower `--rate` or `--max_v` until it does.
- Go slower in tight spaces regardless. Nothing about the policy requires 2 Hz.

`--smooth` is separate: it blends each command with the previous one, so 0.5
means a sharp turn takes two or three decisions to reach full rate. Lower it
toward 0 for gentler motion, raise it toward 1 for a more responsive but
jerkier robot.

---

## 8. Troubleshooting

- `ModuleNotFoundError: numpy._core...` — the PYTHONPATH export was not run in
  this terminal, so conda's numpy 1 is shadowing the venv.
- No camera topic — section 2 step 3. No `/Odometry` — FAST-LIO needs the LiDAR
  running first.
- Robot ignores commands — motion_control is not running, or the robot is not in
  terrain mode.
- First inference takes 300-500 ms — CUDA warm-up. The median settles near 8 ms
  and the node prints it.
- `[CLI default]` in the startup lines — `env_config.json` is missing beside the
  checkpoint.
- Node stuck on "waiting for camera/odom" — one of the two topics is not
  publishing; check with `ros2 topic hz`.
