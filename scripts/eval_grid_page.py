"""One page for the whole eval grid (2026-09-21, Joana: "visualize all of those and the overheads,
clearly organized"). Rewritten 2026-09-22 after two findings: (1) every "goalNN" eval before the
eval fix drew random short goals (those cells are dropped), (2) every policy trained before the image
fix never used its camera (every row is the odometry-only ablation, said so in the banner).

Folder styles read from the Mac evals dir:
  <family>/<ckpt>[-cont]_<scene>_<test>            old style (plaza cells only; corner cells were random goals)
  <family>_cont<N>k/<scene>_<steps>_gfr15-70       continuation plazas
  corner_<family>_<N>k[_jit]/<scene>_<steps>_goal<g>    fixed-goal corner tests (jit = spawn jitter, 8 distinct episodes)
  ped_<family>_<N>k[_BLIND][_jit]/<scene>_<steps>_goal<g>  fixed-goal pedestrian tests
Overheads: pictures/ (plazas) and pictures_0922/ (fixed-goal tests).

    python scripts/eval_grid_page.py <evals_dir> <out.html>
"""
import base64, glob, html, json, os, re, sys

evals, out = sys.argv[1], sys.argv[2]
TESTS = [("quad1_01", "plaza", "quad1_01 plaza, goals ahead 15-70"),
         ("quad1_07", "plaza", "quad1_07 plaza, goals ahead 15-70"),
         ("quad2_00", "goal48", "quad2_00 obstacle corner, spawn 12-22, goal 48"),
         ("sequoia1_17", "goal72", "sequoia1_17 lawn corner, spawn 45-55, goal 72"),
         ("quad2_04", "goal62", "quad2_04 person on the path, spawn 26-30, goal 62"),
         ("sequoia1_10", "goal30", "sequoia1_10 group crossing, spawn 5-7, goal 30"),
         ("packard1_16", "goal25", "packard1_16 person walking past, spawn 4-8, goal 25")]
FAMILY_LABEL = {"A": "A (v33)", "memory": "memory (v33)", "A_v35b": "A (v35b)", "memory_v35b": "memory (v35b)",
                "memory_always": "memory-always (v33)", "veto": "veto (v33)", "map": "map (v33)", "chunk10": "chunk 10 (v33)"}
FAMILY_ORDER = ["A_v35b", "memory", "A", "veto", "memory_v35b", "memory_always", "map", "chunk10"]

def read_cell(f):
    d = json.load(open(f)); eps = d["episodes"]
    n = len(eps); goals = sum(1 for e in eps if e["success"]); crashes = sum(1 for e in eps if e["outcome"] == "CRASH")
    lawn = sum(1 for e in eps if e.get("trespass_steps", 0) > 0); steps = sum(e["steps"] for e in eps) / max(n, 1)
    spawns = len(set(tuple(round(v, 2) for v in e["spawn"][:2]) for e in eps if isinstance(e.get("spawn"), list)))
    return dict(goals=goals, n=n, crashes=crashes, lawn=lawn, steps=steps, distinct=spawns,
                folder=os.path.relpath(os.path.dirname(f), evals))

rows = {}   # (family, steps_k, variant) -> {(scene, test): cell};  variant in {"", "BLIND"}
def put(family, steps_k, variant, scene, test, cell, prefer=False):
    key = (family, steps_k, variant); cur = rows.setdefault(key, {}).get((scene, test))
    if cur is None or prefer or (cell["distinct"] > cur["distinct"]):
        rows[key][(scene, test)] = cell

for f in sorted(glob.glob(os.path.join(evals, "*", "*", "metrics.json"))):
    tag = f.split(os.sep)[-3]; name = f.split(os.sep)[-2]
    if tag in ("scored_with_v14_table", "flawed_goals_behind", "corner_far", "corner_veto_50k", "ped_AV35", "ped_memory",
               "A_MIRRORED", "A_MIRRORED_stamped", "memory_BLIND", "A_v35b_BLIND"):
        continue                                   # random-goal or mirror/blind checks, documented elsewhere
    cell = read_cell(f)
    m = re.match(r"(corner|ped)_(A_v35b|memory_v35b|memory|A|veto)_(\d+)k(_BLIND)?(_jit)?$", tag)
    if m:
        kind, fam, k, blind, jit = m.groups()
        m2 = re.match(r"([a-z0-9]+_\d\d)_(\d+)_goal(\d+)$", name)
        if not m2: continue
        scene, _, g = m2.groups()
        put(fam, int(k), "BLIND" if blind else "", scene, f"goal{g}", cell, prefer=bool(jit))
        if jit: rows[(fam, int(k), "BLIND" if blind else "")][(scene, f"goal{g}")]["jit"] = True
        continue
    m = re.match(r"(A|memory|veto)_cont(\d+)k$", tag)
    if m:
        fam, k = m.groups(); m2 = re.match(r"([a-z0-9]+_\d\d)_(\d+)_gfr15-70$", name)
        if m2: put(fam, int(k), "", m2.group(1), "plaza", cell)
        continue
    m = re.match(r"(\d+)k(?:-cont)?_([a-z0-9]+_\d\d)_(.+)$", name)
    if m and tag in FAMILY_LABEL:
        k, scene, test = m.groups()
        if test.startswith("plaza"): put(tag, int(k), "", scene, "plaza", cell)
        continue

keys = sorted(rows, key=lambda k: (FAMILY_ORDER.index(k[0]) if k[0] in FAMILY_ORDER else 99, k[1], k[2]))

def cell_html(c):
    if c is None: return '<td class="pend">—</td>'
    r = c["goals"] / c["n"]; cls = "g3" if r >= 0.875 else "g2" if r >= 0.6 else "g1" if r >= 0.35 else "g0"
    dist = f' · {c["distinct"]} distinct starts' if c["distinct"] < c["n"] else ""
    return (f'<td class="{cls}"><b>{c["goals"]}/{c["n"]}</b><span class="sub">lawn {c["lawn"]} · crashes {c["crashes"]} · {c["steps"]:.0f} st{dist}{" · jitter" if c.get("jit") else ""}</span></td>')

def img_tag(path, alt):
    if not os.path.exists(path): return f'<p class="miss">missing: {html.escape(os.path.basename(path))}</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'

parts = []
parts.append("""<title>WorldNav Eval Grid</title>
<style>
:root{--bg:#f7f6f2;--ink:#1e2126;--mute:#6b6f76;--line:#d9d6cd;--card:#ffffff;--acc:#2d5b8e;
--g0:#f3d3cf;--g1:#f8e5c6;--g2:#e3ecc9;--g3:#c9e4c5;--pend:#ececec}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--ink:#e8e6e1;--mute:#a1a5ad;--line:#33373d;--card:#1e2126;--acc:#8ab4e8;
--g0:#5a2c28;--g1:#5c4a22;--g2:#3f4f28;--g3:#2b4a2a;--pend:#2a2d33}}
:root[data-theme="dark"]{--bg:#15171a;--ink:#e8e6e1;--mute:#a1a5ad;--line:#33373d;--card:#1e2126;--acc:#8ab4e8;
--g0:#5a2c28;--g1:#5c4a22;--g2:#3f4f28;--g3:#2b4a2a;--pend:#2a2d33}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 28px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 8px;padding-top:14px;border-top:1px solid var(--line)}
p.lead{color:var(--mute);margin:0 0 18px;max-width:70ch}
table{border-collapse:collapse;width:100%;max-width:1100px}th,td{border:1px solid var(--line);padding:6px 8px;text-align:center;vertical-align:top}
th{background:var(--card);font-weight:600}td.arm{text-align:left;white-space:nowrap;background:var(--card)}
td.g0{background:var(--g0)}td.g1{background:var(--g1)}td.g2{background:var(--g2)}td.g3{background:var(--g3)}td.pend{background:var(--pend);color:var(--mute)}
td .sub{display:block;font-size:11px;color:var(--mute);font-variant-numeric:tabular-nums}td.tot{font-weight:700}
.legend{color:var(--mute);font-size:12px;margin:8px 0 0}
figure{margin:12px 0;background:var(--card);border:1px solid var(--line);padding:8px}figure img{max-width:100%;height:auto;display:block}
figcaption{font-size:12px;color:var(--mute);margin-top:6px}
ul.cells{columns:2;column-gap:28px;padding-left:18px;max-width:1100px}ul.cells li{margin:2px 0;font-size:12.5px;break-inside:avoid}code{font-size:12px}
p.miss{color:var(--mute);font-size:12px}
</style>
<h1>WorldNav Eval Grid</h1>
<p class="lead"><b>Every policy on this page was trained before the image fix of 2026-09-22 and never used its camera</b> (the image reached DINOv2 divided by 255 twice; blind and sighted runs agree to 4 mm). These rows are the odometry-only ablation. Evaluated in the generated world with the training traversability table (grass 0.0), 8 episodes per cell, v33 semantics for the v33 arms and v35b epoch 3 for the v35b arms. A cell shows goals reached, then episodes that touched lawn, crashes, and the mean episode length in steps.</p>""")

parts.append('<table><tr><th>policy (checkpoint)</th>' + "".join(f"<th>{html.escape(t[2])}</th>" for t in TESTS) + '</tr>')
for fam, k, var in keys:
    r = rows[(fam, k, var)]; cells = [r.get((t[0], t[1])) for t in TESTS]
    label = f'{FAMILY_LABEL.get(fam, fam)} {k}k' + (" · BLIND (image zeroed)" if var else "")
    parts.append(f'<tr><td class="arm">{html.escape(label)}</td>' + "".join(cell_html(c) for c in cells) + '</tr>')
parts.append('</table><p class="legend">Color: goals reached, dark green 7-8, light green 5-6, tan 3-4, red 0-2. Corner and pedestrian cells are fixed-goal tests (the goal is the same recorded frame in every episode); "distinct starts" says how many different spawns the 8 episodes had when no spawn jitter was used; "jitter" marks the reruns with ±10° / ±0.3 m spawn jitter. The old corner cells (random short goals) are dropped.</p>')

pics, pics2 = os.path.join(evals, "pictures"), os.path.join(evals, "pictures_0922")
OVERHEAD = {("quad1_01", "plaza"): [(pics2, "plaza_quad1_01.png", "memory 150k | veto 100k | A 240k"), (pics, "arms_tw_quad1_01.png", "A 60k | memory 50k | A 110k | memory 70k (v33)"), (pics, "arms2_quad1_01.png", "memory v35b 40k | A v35b 70k | memory-always 50k | veto 50k")],
            ("quad1_07", "plaza"): [(pics2, "plaza_quad1_07.png", "memory 150k | veto 100k | A 240k"), (pics, "arms_tw_quad1_07.png", "A 60k | memory 50k | A 110k | memory 70k (v33)"), (pics, "arms2_quad1_07.png", "memory v35b 40k | A v35b 70k | memory-always 50k | veto 50k")],
            ("quad2_00", "goal48"): [(pics2, "corner_quad2_00.png", "fixed goal 48: A v35b 70k | memory 50k | veto 100k | memory 150k | A 110k | A 240k — failures crash AT the goal, which sits against the wall")],
            ("sequoia1_17", "goal72"): [(pics2, "corner_sequoia1_17.png", "fixed goal 72: every policy cuts straight across the lawn")],
            ("quad2_04", "goal62"): [(pics2, "ped_quad2_04.png", "A v35b 70k | memory 50k")],
            ("sequoia1_10", "goal30"): [(pics2, "ped_sequoia1_10.png", "A v35b 70k sighted | blind | memory 50k sighted | blind")],
            ("packard1_16", "goal25"): [(pics2, "ped_packard1_16.png", "A v35b 70k sighted | blind | memory 50k sighted | blind")]}
for scene, test, title in TESTS:
    parts.append(f"<h2>{html.escape(title)}</h2>")
    for d, fn, cap in OVERHEAD.get((scene, test), []):
        parts.append(img_tag(os.path.join(d, fn), f"{scene}: {cap}"))
    items = []
    for fam, k, var in keys:
        c = rows[(fam, k, var)].get((scene, test))
        if c: items.append(f"<li><b>{html.escape(FAMILY_LABEL.get(fam, fam))} {k}k{' BLIND' if var else ''}</b>: {c['goals']}/{c['n']} goals, lawn {c['lawn']}, crashes {c['crashes']}, {c['distinct']} distinct starts — <code>{html.escape(c['folder'])}</code></li>")
    parts.append('<p class="legend">Videos: campus_data_2026-09-19/evals/ then the folder named on each line.</p><ul class="cells">' + "".join(items) + "</ul>")
open(out, "w").write("\n".join(parts)); print("wrote", out, f"{os.path.getsize(out)/1e6:.1f} MB", len(keys), "rows")
