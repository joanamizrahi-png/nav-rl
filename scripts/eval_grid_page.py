"""One page for the whole eval grid (2026-09-21, Joana: "visualize all of those and the overheads,
clearly organized, because the numbers are not always consistent with what we see").

Reads the Mac eval folder (evals/<arm>/<ckpt>_<scene>_<test>/metrics.json) and the overhead pictures
(evals/pictures/*.png), and writes a single self-contained HTML page: the grid of goals / lawn /
phantom per arm x test, then one section per test with the overheads and the folder of every cell
so the videos can be opened. Pictures are embedded as data URIs. numpy-free, stdlib only.

    python scripts/eval_grid_page.py <evals_dir> <out.html>
"""
import base64, glob, html, json, os, re, sys

evals, out = sys.argv[1], sys.argv[2]
TESTS = [("quad1_01", "plaza_goals-ahead-frames15-70", "quad1_01 plaza, goals ahead"),
         ("quad1_07", "plaza_goals-ahead-frames15-70", "quad1_07 plaza, goals ahead"),
         ("quad2_00", "corner_spawn12-22_goal48", "quad2_00 corner, spawn 12-22, goal 48"),
         ("sequoia1_17", "corner_spawn45-55_goal72", "sequoia1_17 corner, spawn 45-55, goal 72")]
FAR = ("quad2_00", "corner-far_spawn5-11_goal48", "quad2_00 far corner, spawn 5-11, goal 48")
ARM_ORDER = ["A", "memory", "A_v35b", "memory_v35b", "memory_always", "veto", "map", "chunk10", "memory_BLIND", "A_v35b_BLIND", "A_MIRRORED"]
ARM_LABEL = {"A": "A (v33)", "memory": "memory (v33)", "A_v35b": "A (v35b)", "memory_v35b": "memory (v35b)",
             "memory_always": "memory-always (v33)", "veto": "veto (v33, veto on)", "map": "map (v33)", "chunk10": "chunk 10 (v33)", "memory_BLIND": "memory (v33), BLIND: image zeroed", "A_v35b_BLIND": "A (v35b), BLIND: image zeroed", "A_MIRRORED": "A (v33), every episode MIRRORED"}

rows = {}  # (arm, ckpt) -> {test: cell}
for f in sorted(glob.glob(os.path.join(evals, "*", "*", "metrics.json"))):
    arm = f.split(os.sep)[-3]; name = f.split(os.sep)[-2]
    if arm in ("scored_with_v14_table", "flawed_goals_behind"): continue
    m = re.match(r"(\d+k(?:-cont)?)_(quad\d_\d\d|sequoia\d_\d\d)_(.+)$", name)
    if not m: continue
    ck, scene, test = m.groups()
    d = json.load(open(f)); s = d["summary"]; eps = d["episodes"]
    n = len(eps); goals = sum(1 for e in eps if e["success"]); crashes = sum(1 for e in eps if e["outcome"] == "CRASH")
    ph = sum(1 for e in eps if e.get("crash_was_phantom")); lawn = sum(1 for e in eps if e.get("trespass_steps", 0) > 0)
    steps = sum(e["steps"] for e in eps) / max(n, 1)
    rows.setdefault((arm, ck), {})[(scene, test)] = dict(goals=goals, n=n, crashes=crashes, phantom=ph, lawn=lawn, steps=steps,
                                                        folder=os.path.relpath(os.path.dirname(f), evals))

def ck_key(ck):
    v = int(ck.replace("k", "").replace("-cont", "")); return v + (0.5 if "cont" in ck else 0)
keys = sorted(rows, key=lambda k: (ARM_ORDER.index(k[0]) if k[0] in ARM_ORDER else 99, ck_key(k[1])))

def cell_html(c):
    if c is None: return '<td class="pend">pending</td>'
    r = c["goals"] / c["n"]; cls = "g3" if r >= 0.875 else "g2" if r >= 0.6 else "g1" if r >= 0.35 else "g0"
    return (f'<td class="{cls}"><b>{c["goals"]}/{c["n"]}</b><span class="sub">lawn {c["lawn"]} · phantom {c["phantom"]}/{c["crashes"]} · {c["steps"]:.0f} st</span></td>')

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
<p class="lead">Every policy checkpoint evaluated in the generated world with the training traversability table (grass 0.0), 8 episodes per cell, goals ahead of the spawn, v33 semantics for the v33 arms and v35b epoch 3 for the v35b arms. A cell shows goals reached, then the number of episodes that touched lawn, the phantom crashes over all crashes, and the mean episode length in steps.</p>""")
parts.append('<table><tr><th>arm (checkpoint)</th>' + "".join(f"<th>{html.escape(t[2])}</th>" for t in TESTS) + '<th>total /32</th><th>far corner</th></tr>')
for arm, ck in keys:
    r = rows[(arm, ck)]; cells = [r.get((t[0], t[1])) for t in TESTS]
    tot = sum(c["goals"] for c in cells if c); n = sum(c["n"] for c in cells if c)
    far = r.get((FAR[0], FAR[1]))
    parts.append(f'<tr><td class="arm">{html.escape(ARM_LABEL.get(arm, arm))} {html.escape(ck)}</td>' + "".join(cell_html(c) for c in cells)
                 + f'<td class="tot">{tot}/{n}' + (" (partial)" if n < 32 else "") + '</td>' + (cell_html(far) if far else '<td class="pend">—</td>') + '</tr>')
parts.append('</table><p class="legend">Color: goals reached, dark green 7-8, light green 5-6, tan 3-4, red 0-2. "phantom a/b": a of the b crashes were labels the fused map calls walkable. Phantom counts are honest for every arm except "veto on", whose surviving crashes are by construction the non-phantom ones.</p>')

pics = os.path.join(evals, "pictures")
for scene, test, title in TESTS + [FAR]:
    parts.append(f"<h2>{html.escape(title)}</h2>")
    if (scene, test) == (FAR[0], FAR[1]):
        parts.append(img_tag(os.path.join(pics, "arms_tw_quad2_00_far.png"), "far corner: memory 50k (left), A 60k (right); v33 semantics"))
    else:
        parts.append(img_tag(os.path.join(pics, f"arms_tw_{scene}.png"), f"{scene}: v33 arms — A 60k | memory 50k | A continued 110k | memory continued 70k"))
        parts.append(img_tag(os.path.join(pics, f"arms2_{scene}.png"), f"{scene}: new arms — memory on v35b 40k | A on v35b 70k | memory-always 50k | veto 50k (veto on); missing panels = cell still running"))
    items = []
    for arm, ck in keys:
        c = rows[(arm, ck)].get((scene, test))
        if c: items.append(f"<li><b>{html.escape(ARM_LABEL.get(arm, arm))} {html.escape(ck)}</b>: {c['goals']}/{c['n']} goals, lawn {c['lawn']}, phantom {c['phantom']}/{c['crashes']} — <code>{html.escape(c['folder'])}</code></li>")
    parts.append('<p class="legend">Videos: campus_data_2026-09-19/evals/ then the folder named on each line; failures/ inside holds the collision pictures.</p><ul class="cells">' + "".join(items) + "</ul>")
open(out, "w").write("\n".join(parts)); print("wrote", out, f"{os.path.getsize(out)/1e6:.1f} MB", len(keys), "arm rows")
