# Project status — 2026-08-13

One page. What works, what the numbers are, what is decided, what is next.
Replaces PLAN.md as the working reference (PLAN.md kept as history).

## What works, with its number

| System | State | Key number |
|---|---|---|
| World model (reconstruct + render) | validated | replay ≈ original video; ground ~2.7 cm; trust region ±1 m / ±45° |
| RL policy, single scene | **closed** | 100% success at all 9 goal distances, zero-shot **100% on 7 unseen scenes** |
| RL policy, multi-scene | closed by transfer | joint training unstable in both KL regimes (documented negative result); unnecessary |
| Semantic model v10 | **superseded v9** | held-out **79.9% / 80.6%** with the reader decode; own RGB at parity with vanilla vs real footage |
| Semantic model v11 (final) | trained, grading today | ablation winners composed (leash 0.3 + big reader); targets ≥82 / ≥80 |
| Class taxonomy | **approved** | 14 classes, single source of truth module |
| Reward | audited | terminal bonus ≈70% of incentive; per-step economy terrain-dominated 2:1 |
| Pose→image direction | **bug confirmed, fix planned** | yaw is mirrored between pose math and render (left-handed nav frame), both scenes tested. Sim self-consistent → all results stand; fix at calibration level rides with the v14 policy retrain, before any robot work |

## Decisions made this week (and why)

- **One-pass inference**: the RGB-preservation loss closed the finetune's RGB
  gap (fidelity to real frames now equals the vanilla model: 13.7 vs 13.2 dB /
  18.7 vs 19.2 dB), so RGB + semantics come from a single pass, aligned by
  construction. The two-pass + anchoring design is retired to
  methodology/ablation status — it proved one-pass safe.
- **Reader decode replaces palette-snap** (reverses the v9-era decision):
  with CE at all timesteps the co-trained reader wins decisively
  (79.9/80.6 vs 70.0/54.8 on v10). Model + reader are a matched pair.
- **Every training ingredient attributed** (5-epoch smokes, then 30-epoch
  ablations): all-timestep CE +4.4, preservation leash 0.3 > 1.0 for accuracy
  (82.3 vs 79.9) with the RGB win kept, min-SNR +6.1 alone, bigger reader
  head +1.1. v11 composes the winners.
- **Held-out clips are permanent**: rugd_trail-6_01, rugd_park-1_02 never
  enter training.
- **PPO protections**: unchanged (checkpoint-every-2k + eval-the-peak).

## Open items

1. **Yaw mirror fix** — make the nav frame right-handed at the calibration
   level (one place: env motion, obs, overlays all inherit), re-probe both
   scenes, replay must still pass; absorbed by the v14 policy retrain.
2. **v11 grades** — accuracy both clips (reader), drift, fidelity. If passed,
   v11 is the paper's semantic model.
3. **Encoder experiment (analog bits) & VGGT accuracy report card** — scoped,
   queued behind the yaw fix (see NeoVerse/docs/ENCODER_VS_ROBOT.md for the
   robot-first recommendation).
4. **v14 reward switch + policy retrain**, then **Go2W deployment** — the
   remaining chapter. Training freeze ~Aug 24.

## Where everything lives

- Results for humans: `World Model/Drive_package/` (drag to Google Drive).
- All fetched results: `inference_runs/` — model renders by version
  (`v9_*, v10_*, v10b..e_*, v11_*`), judgment videos in `V10_verdict/`,
  policy videos in `rl_rollouts/`, probes in `diagnostics/`,
  held-out evidence in `HELDOUT_v9/`.
- Docs: `nav-rl/ARCHITECTURE.md` (systems + all losses, current),
  `WORLD_MODEL_LIMITS.md`, `NeoVerse/docs/CLASS_SET_PROPOSAL.md`,
  `NeoVerse/docs/ENCODER_VS_ROBOT.md` (the pending decision),
  `NeoVerse/docs/SEMANTIC_V8_RESEARCH.md`.
- One-page web summary with diagrams: artifact link in chat
  (pipeline + losses + version table).
- Single source of truth for classes: `NeoVerse/diffsynth/utils/class_taxonomy.py`.
