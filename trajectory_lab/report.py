"""Portable HTML report with no CDN, JavaScript, or external assets."""

import html
import json


def render(data):
    esc = lambda v: html.escape("null" if v is None else str(v))
    cards = "".join(f'<div class="card"><strong>{esc(data[k])}</strong><span>{label}</span></div>' for k, label in
                    (("trajectories", "Trajectories"), ("groups", "Rollout groups"), ("errors", "Errors"), ("warnings", "Warnings")))
    findings = "".join(f'<tr><td class="{f["severity"]}">{esc(f["severity"])}</td><td><code>{esc(f["code"])}</code></td><td>{esc(f["item"])}</td><td>{esc(f["detail"])}</td></tr>' for f in data["findings"])
    evaluations = "".join('<tr>' + ''.join(f'<td>{esc(e[k])}</td>' for k in ("run_id", "policy_version", "split", "source", "tasks", "k", "pass_at_k")) + '</tr>' for e in data["evaluations"])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Trajectory Lab — Audit</title><style>
body{margin:0;background:#f5f3ee;color:#172c35;font:16px/1.65 system-ui,sans-serif}main{max-width:1150px;margin:48px auto;padding:0 24px}h1{font-size:42px;line-height:1.1;margin:12px 0}h2{margin-top:36px}small,.muted{color:#52666a}.eyebrow{letter-spacing:.15em;text-transform:uppercase;color:#227365;font-weight:700}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}.card{background:white;border:1px solid #dbe1dd;border-radius:12px;padding:20px}.card strong{display:block;font-size:36px}.card span{color:#52666a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #d4ddd8;padding:12px 10px;text-align:left;vertical-align:top}th{background:#e7ede8}.scroll{overflow:auto}.error{color:#ad3529}.warning{color:#865e12}code{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e7ede8;padding:20px;border-radius:10px}footer{margin:40px 0;color:#52666a}@media(max-width:650px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:32px}}</style>
<main><div class="eyebrow">Agent RL · Inspect before training</div><h1>Trajectory audit</h1>
<p class="muted">Mask alignment, group integrity, split overlap and evaluation accounting.</p>
<p><strong>Read the source column:</strong> synthetic fixtures demonstrate checks, not model performance.</p>
<div class="cards">''' + cards + '''</div><h2>Findings</h2><div class="scroll"><table><tr><th>Severity</th><th>Check</th><th>Item</th><th>Explanation</th></tr>''' + findings + '''</table></div>
<h2>Evaluation strata</h2><p>Each policy / run / split / source is evaluated separately. Null means unavailable.</p><div class="scroll"><table><tr><th>Run</th><th>Policy</th><th>Split</th><th>Source</th><th>Tasks</th><th>k</th><th>pass@k</th></tr>''' + evaluations + '''</table></div>
<h2>Interpretation limits</h2><ul>''' + ''.join(f'<li>{esc(x)}</li>' for x in data['caveats']) + '''</ul><details><summary>Complete report and provenance</summary><pre>''' + esc(json.dumps(data, indent=2, ensure_ascii=False)) + '''</pre></details><footer>Agent Trajectory Lab · v0.1.0 · CPU only, zero runtime dependencies</footer></main></html>'''
