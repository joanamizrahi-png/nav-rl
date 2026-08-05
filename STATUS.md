# Project status — 2026-08-05

One page. What works, what the numbers are, what is decided, what is next.
Replaces PLAN.md as the working reference (PLAN.md kept as history).

## What works, with its number

| System | State | Key number |
|---|---|---|
| World model (reconstruct + render) | validated | replay ≈ original video; ground ~2.7 cm; trust region ±1 m / ±45° |
| RL policy, single scene | **closed** | 100% success at all 9 goal distances, near-shortest paths (5–21 steps), zero-shot transfer to 2 unseen scenes |
| RL policy, multi-scene | in progress | v7 strangled by KL leash (4,837 truncated updates), peak rescued at 60%; **7b (unleashed) training now** |
| Semantic model v9 (14 classes) | trained | **held-out 78.8% / 69.9% pixel accuracy** — first honest evaluation; trail/grass/obstacle 57–81 IoU |
| Class taxonomy | proposed | 14 classes, single source of truth module, id-order bug retired; **awaiting advisor** |
| Reward | audited | terminal bonus ≈70% of incentive; per-step economy terrain-dominated 2:1 |

## Decisions made (and why)

- **Reward reads rasterized labels; diffused labels enter only as verified**
  (held-out eval defines the trust frontier; alpha marks invented pixels).
- **Palette-snap stays the decode**: the learned reader was tested head-to-head
  on held-out scenes and lost (73.2 vs 78.8). Polish path is capped; if the
  bar demands more, the SAM3-encoder (U4) is the remaining lever.
- **Two-pass inference** (vanilla RGB / finetuned semantics) for deployment;
  the doubled diffusion cost never touches the RL loop (ribbon cache is the
  leading candidate for diffused observations — mechanism decision open).
- **PPO protections**: lr decay and KL leash both tested and rejected with
  evidence; checkpoint-every-2k + eval-the-peak is the collapse defense.
- **Held-out clips are permanent**: rugd_trail-6_01, rugd_park-1_02 never
  enter training again.

## Open decisions

1. **Class set** — advisor sign-off on the 14-class proposal (two flagged
   judgment calls: pavement-unknown default, sand=rough). Gates the final
   semantic training.
2. **U4 (SAM3-encoder label pathway)** — build (4–6 d) or accept ~75–79%
   held-out. Decide with advisor, baseline in hand.
3. **Diffused-observations mechanism** — precomputed ribbon cache vs
   live-every-Nth. Decide before the diffused-obs training run.

## This week's remaining items

- Read 7b (multi-scene) + checkpoint sweep — tonight.
- Vanilla-vs-finetune off-trajectory comparison — renders in queue.
- Drive upload + share (package ready: 188 artifacts + READMEs).
- Message to advisor: proposal + Drive link + held-out numbers + meeting ask.

## Next week's queue (effort)

| Item | Effort |
|---|---|
| Conservative/aggressive reward knob + demo pair | ~0.5 d |
| v14 switch for RL (labels_v14 + yaml_v14 + retrain) | ~0.5 d + 1 run |
| Dynamic-objects scene (clip selection + env stage) | ~1–2 d |
| Constant-offset trajectory variant (policy-like camera) | ~0.5 d |
| 360°/wider-coverage data plan | scoping |
| U4 if approved | 4–6 d |

## Where everything lives

- Results for humans: `World Model/Drive_package/` (drag to Google Drive).
- Held-out evidence: `inference_runs/HELDOUT_v9/` (videos + README + legend).
- Docs: `nav-rl/ARCHITECTURE.md` (systems + losses), `WORLD_MODEL_LIMITS.md`
  (trust boundaries), `NeoVerse/docs/CLASS_SET_PROPOSAL.md` (taxonomy),
  `NeoVerse/docs/SEMANTIC_V8_RESEARCH.md` (literature).
- Single source of truth for classes: `NeoVerse/diffsynth/utils/class_taxonomy.py`.
- Diagrams: `nav-rl/diagrams/` + the web version (artifact link in chat).
