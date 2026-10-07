# MIID/benchmark/report.py
#
# Self-contained HTML report: each round as a strip of the base face and its
# variations, every image badged with the predicted score and the reason.
# Thumbnails are inlined so the file can be copied off the box and opened
# anywhere.

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

THUMB_HEIGHT = 260

_CSS = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5d6678; --line:#dfe3ea;
  --good:#1f8a4c; --ok:#6b8f1f; --mid:#b7791f; --bad:#c0392b; --worst:#7b1f1f; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171d; --card:#1d2129; --ink:#e7eaf0;
  --muted:#9aa3b5; --line:#2c323d; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink);
  font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:0; }
.muted { color:var(--muted); }
.summary, .round { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px; margin:0 auto 16px; max-width:1500px; }
.stats { display:flex; flex-wrap:wrap; gap:24px; margin-top:10px; }
.stat b { display:block; font-size:22px; }
table { border-collapse:collapse; margin-top:12px; font-size:13px; }
td, th { padding:4px 10px; border-bottom:1px solid var(--line); text-align:right; }
td:first-child, th:first-child { text-align:left; }
.strip { display:flex; gap:12px; overflow-x:auto; padding:10px 0 4px; }
.tile { flex:0 0 auto; width:210px; }
.tile img { width:210px; height:auto; border-radius:6px; display:block; border:1px solid var(--line); }
.badge { display:inline-block; min-width:30px; padding:1px 8px; border-radius:999px; color:#fff;
  font-weight:600; text-align:center; margin-right:6px; }
.s5 { background:var(--good); } .s4, .s3 { background:var(--ok); } .s2, .s1, .s0 { background:var(--mid); }
.sn1, .sn2 { background:var(--bad); } .sn3, .sn4, .sn5 { background:var(--worst); }
.cap { font-size:12px; margin-top:6px; } .cap .type { font-weight:600; }
.why { color:var(--muted); font-size:12px; margin-top:2px; }
.warn { color:var(--mid); font-size:12px; margin-top:2px; }
"""


def _thumb(path: str) -> str:
    try:
        img = Image.open(path).convert("RGB")
    except OSError:
        return ""
    w = int(img.width * THUMB_HEIGHT / img.height)
    img = img.resize((w, THUMB_HEIGHT), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _badge(score: int) -> str:
    cls = f"s{score}" if score >= 0 else f"sn{-score}"
    return f'<span class="badge {cls}">{score:+d}</span>'


def write_html(path: Path, rounds: List[Dict[str, Any]], run: Dict[str, Any]) -> None:
    e = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Submission Benchmark</title>",
        f"<style>{_CSS}</style></head><body>",
    ]

    if run:
        kav = sum(r["summary"]["expected_gated_norm"] for r in rounds) / max(len(rounds), 1)
        rep = sum(r["summary"]["delta_rep"] for r in rounds) / max(len(rounds), 1)
        parts.append("<section class='summary'><h1>Predicted grading of archived submissions</h1>")
        parts.append("<div class='muted'>Predictions from the published score sheet plus "
                     "locally calibrated detectors — not the grader's output.</div>")
        parts.append("<div class='stats'>")
        for label, value in (
            ("variations", f"{run['n']}"),
            ("rounds", f"{len(rounds)}"),
            ("mean predicted score", f"{run['mean_score']:+.2f}"),
            ("expected KAV / round", f"{kav:.3f}"),
            ("Δrep / round", f"{rep:+.3f}"),
        ):
            parts.append(f"<div class='stat'><b>{e(value)}</b><span class='muted'>{e(label)}</span></div>")
        parts.append("</div>")
        for title, key in (("By model", "by_model"), ("By slot", "by_slot")):
            parts.append(f"<table><tr><th>{title}</th><th>n</th><th>mean</th><th>Δrep</th>"
                         "<th>=5</th><th>&lt;0</th></tr>")
            for name, s in run[key].items():
                parts.append(
                    f"<tr><td>{e(name)}</td><td>{s['n']}</td><td>{s['mean_score']:+.2f}</td>"
                    f"<td>{s['mean_delta_rep']:+.3f}</td><td>{s['share_5'] * 100:.0f}%</td>"
                    f"<td>{s['share_negative'] * 100:.0f}%</td></tr>"
                )
            parts.append("</table>")
        parts.append("<table><tr><th>Most common reasons</th><th>n</th></tr>")
        for reason, count in run["top_reasons"]:
            parts.append(f"<tr><td>{e(reason)}</td><td>{count}</td></tr>")
        parts.append("</table></section>")

    for rnd in reversed(rounds):
        s = rnd["summary"]
        parts.append("<section class='round'>")
        parts.append(
            f"<h2>{e(rnd['record_id'])}</h2><div class='muted'>model {e(str(rnd.get('model')))}"
            f" · validator {e(str(rnd.get('validator')))} · expected KAV {s['expected_gated_norm']:.2f}"
            f" · Δrep {s['delta_rep']:+.2f}</div><div class='strip'>"
        )
        if rnd.get("base"):
            parts.append(f"<div class='tile'><img src='{_thumb(rnd['base'])}' alt='base'>"
                         "<div class='cap'><span class='type'>base image</span></div></div>")
        for item in rnd["items"]:
            res = item["result"]
            why = res["reasons"][0] if res["reasons"] else ""
            ident = "-" if res["identity"] is None else f"{res['identity']:.3f}"
            parts.append(f"<div class='tile'><img src='{_thumb(item['image'])}' alt=''>")
            parts.append(
                f"<div class='cap'>{_badge(res['score'])}<span class='type'>"
                f"{e(item['variation_type'])}</span> <span class='muted'>{e(item['intensity'])}"
                f" · id {ident}</span></div><div class='why'>{e(why)}</div>"
            )
            for extra in res["reasons"][1:3]:
                parts.append(f"<div class='why'>{e(extra)}</div>")
            for w in res["warnings"][:2]:
                parts.append(f"<div class='warn'>{e(w)}</div>")
            parts.append("</div>")
        for missing in rnd.get("missing_slots") or []:
            parts.append(f"<div class='tile'><div class='cap'>{_badge(0)}<span class='type'>"
                         f"{e(missing)}</span></div><div class='why'>not submitted</div></div>")
        parts.append("</div></section>")

    parts.append("</body></html>")
    path.write_text("".join(parts))
