# MIID/benchmark/runner.py
#
# Orchestration for the CLI: find records, measure (with a per-record cache),
# score, print, and optionally write JSON / HTML. Imported only after
# __main__ has pinned the device, because analyzers pull in torch lazily.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MIID.benchmark import scoring
from MIID.benchmark.analyzers import ANALYZER_VERSION, Analyzer, gender_from_filename, sha256_file
from MIID.benchmark.loader import Round, VariationItem, find_record_dirs, load_record, single_item

CACHE_NAME = "benchmark.json"


def _default_archive() -> Path:
    # Same resolution the miner uses, without importing the miner package.
    import os

    override = os.environ.get("MIID_REQUEST_ARCHIVE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "miner_requests"


# ── cache ────────────────────────────────────────────────────────────────


def _load_cache(record_dir: Optional[Path], use_clip: bool) -> Dict[str, Any]:
    if record_dir is None:
        return {}
    try:
        data = json.loads((record_dir / CACHE_NAME).read_text())
    except (OSError, ValueError):
        return {}
    if data.get("analyzer_version") != ANALYZER_VERSION:
        return {}
    if use_clip and not data.get("clip"):
        return {}
    return data.get("items") or {}


def _save_cache(record_dir: Optional[Path], items: Dict[str, Any], use_clip: bool) -> None:
    if record_dir is None:
        return
    payload = {"analyzer_version": ANALYZER_VERSION, "clip": use_clip, "items": items}
    try:
        (record_dir / CACHE_NAME).write_text(json.dumps(payload, default=_json_default))
    except OSError as e:
        print(f"  (could not write cache in {record_dir}: {e})", file=sys.stderr)


def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return str(obj)


# ── measuring ────────────────────────────────────────────────────────────


def _measure(analyzer: Analyzer, item: VariationItem, cache: Dict[str, Any], force: bool) -> Dict[str, Any]:
    key = item.image.name
    cached = cache.get(key)
    if cached and not force:
        if cached.get("variation", {}).get("sha256") == sha256_file(item.image):
            return cached
    metrics = analyzer.pair_metrics(
        item.base,
        item.image,
        background_slot=item.is_background,
        accessory_requested=bool(item.accessory),
        base_gender=gender_from_filename(item.base_filename),
    )
    metrics = json.loads(json.dumps(metrics, default=_json_default))
    cache[key] = metrics
    return metrics


def _wanted(item: VariationItem, include_dropped: bool) -> bool:
    return include_dropped or item.status in (None, "submitted")


# ── printing ─────────────────────────────────────────────────────────────


def _fmt_id(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def _print_round(rnd: Round, scored: List[Tuple[VariationItem, scoring.ScoreResult]],
                 summary: Dict[str, Any], verbose: bool) -> None:
    head = f"{rnd.record_id}"
    meta = [f"model={rnd.model}" if rnd.model else None,
            f"validator={rnd.validator}" if rnd.validator else None]
    print(f"\n{head}  {'  '.join(m for m in meta if m)}")
    print(f"  {'#':>2} {'variation':31} {'intensity':13} {'score':>5} {'identity':>8} {'Δrep':>6}  why")
    for item, res in scored:
        why = res.reasons[0] if res.reasons else ""
        if len(res.reasons) > 1:
            why += f"  (+{len(res.reasons) - 1} more)"
        dropped = "  [not submitted]" if item.status == "dropped" else ""
        print(f"  {item.slot:>2} {item.variation_type:31} {item.intensity:13} {res.score:>5} "
              f"{_fmt_id(res.identity):>8} {res.delta_rep:>+6.2f}  {why}{dropped}")
        if verbose:
            for reason in res.reasons[1:]:
                print(f"  {'':>63}  · {reason}")
            for d in res.detections:
                value = "" if d.value is None else f" ({d.value:.2f})"
                print(f"  {'':>63}  ~ {d.component}: wanted {d.requested}, measured {d.detected}{value} -> {d.match}")
            if res.extras:
                print(f"  {'':>63}  ~ unrequested: {', '.join(res.extras)}")
            if res.accessory:
                acc = res.accessory
                style = f", style {acc['covering_style']}" if acc.get("covering_style") else ""
                print(f"  {'':>63}  ~ accessory: {acc.get('status')} — top {acc.get('top')} "
                      f"({acc.get('top_confidence', 0):.2f}){style}")
        for w in res.warnings:
            print(f"  {'':>63}  ! {w}")
    for missing in rnd.missing_slots:
        print(f"  {'-':>2} {missing:31} {'':13} {'0':>5} {'  -  ':>8} {0:>+6.2f}  not submitted (no image) — scored 0")
    print(f"  expected KAV this round {summary['expected_gated_norm']:.2f} "
          f"(validation {summary['expected_validation_norm']:.2f}, identity gate "
          f"{summary['identity_gate_pass_rate'] * 100:.0f}% of slots) | Δrep if reviewed {summary['delta_rep']:+.2f}")


def _print_run(run: Dict[str, Any], rounds: List[Dict[str, Any]]) -> None:
    if not run:
        return
    print("\n" + "=" * 100)
    print(f"{run['n']} variations over {len(rounds)} round(s)")
    if rounds:
        kav = sum(r["summary"]["expected_gated_norm"] for r in rounds) / len(rounds)
        rep = sum(r["summary"]["delta_rep"] for r in rounds) / len(rounds)
        print(f"  mean expected KAV per round  {kav:.3f}  (1.0 = every slot a 5 with identity >= 0.6)")
        print(f"  mean Δrep per round          {rep:+.3f}  (x rounds/day for the reputation trend)")
    print(f"  mean predicted score         {run['mean_score']:+.2f}")
    hist = run["histogram"]
    print("  score histogram   " + "  ".join(f"{s:+d}:{hist[s]}" for s in range(-5, 6) if hist[s]))

    def table(title, rows):
        print(f"\n  {title:28} {'n':>4} {'mean':>6} {'Δrep':>7} {'=5':>5} {'<0':>5}")
        for key, s in rows.items():
            print(f"  {key:28} {s['n']:>4} {s['mean_score']:>+6.2f} {s['mean_delta_rep']:>+7.3f} "
                  f"{s['share_5'] * 100:>4.0f}% {s['share_negative'] * 100:>4.0f}%")

    table("by model", run["by_model"])
    table("by slot", run["by_slot"])
    print("\n  most common reasons")
    for reason, count in run["top_reasons"]:
        print(f"  {count:>4}  {reason}")


# ── entry ────────────────────────────────────────────────────────────────


def run(args) -> int:
    try:
        import torch

        torch.set_num_threads(max(1, args.threads))
    except ImportError:
        pass

    analyzer = Analyzer(device=args.device, use_clip=not args.no_clip)
    started = time.time()
    rounds_out: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    try:
        if args.image or args.base:
            if not (args.image and args.base and args.var_type):
                print("--base, --image and --type are all required for a single image", file=sys.stderr)
                return 2
            item = single_item(Path(args.base), Path(args.image), args.var_type,
                               args.intensity, args.detail)
            rounds = [Round("adhoc", None, None, None, None, 1, [item], [])]
        else:
            roots = [Path(p) for p in args.paths] or [_default_archive()]
            record_dirs = find_record_dirs(roots)
            if args.last:
                record_dirs = record_dirs[-args.last:]
            rounds = []
            skipped: Dict[str, int] = {}
            for record_dir in record_dirs:
                rnd, why = load_record(record_dir)
                if rnd is None:
                    skipped[why] = skipped.get(why, 0) + 1
                else:
                    rounds.append(rnd)
            if skipped:
                for why, n in skipped.items():
                    print(f"skipped {n} record(s): {why}")
            if not rounds:
                print(f"No benchmarkable records under {', '.join(map(str, roots))}. "
                      "Records need media/ — run the miner with MIID_ARCHIVE_IMAGES=1.")
                return 1

        measured: List[Tuple[Round, List[Tuple[VariationItem, Dict[str, Any]]]]] = []
        for rnd in rounds:
            cache = _load_cache(rnd.record_dir, analyzer.use_clip)
            pairs = []
            for item in rnd.items:
                if not _wanted(item, args.include_dropped):
                    continue
                pairs.append((item, _measure(analyzer, item, cache, args.force)))
            _save_cache(rnd.record_dir, cache, analyzer.use_clip)
            measured.append((rnd, pairs))

        labels = [(f"{rnd.record_id}#{item.slot}", m) for rnd, pairs in measured for item, m in pairs]
        duplicates = scoring.find_duplicates(labels)

        for rnd, pairs in measured:
            scored = []
            for item, m in pairs:
                res = scoring.score_variation(item, m, duplicates.get(f"{rnd.record_id}#{item.slot}"))
                scored.append((item, res))
                rows.append({
                    "record_id": rnd.record_id,
                    "slot": item.slot,
                    "slot_type": item.variation_type,
                    "intensity": item.intensity,
                    "model": item.model,
                    "status": item.status,
                    "image": str(item.image),
                    "base": str(item.base),
                    "score": res.score,
                    "reasons": res.reasons,
                    "result": res.to_dict(),
                })
            submitted = [res for item, res in scored if item.status != "dropped"]
            summary = scoring.summarize_round(rnd.requested, submitted, len(rnd.missing_slots))
            _print_round(rnd, scored, summary, args.verbose)
            rounds_out.append({
                "record_id": rnd.record_id,
                "started_at": rnd.started_at,
                "model": rnd.model,
                "validator": rnd.validator,
                "summary": summary,
                "missing_slots": rnd.missing_slots,
                "base": str(rnd.items[0].base) if rnd.items else None,
                "items": [
                    {
                        "slot": item.slot,
                        "variation_type": item.variation_type,
                        "intensity": item.intensity,
                        "detail": item.detail,
                        "status": item.status,
                        "model": item.model,
                        "image": str(item.image),
                        "result": res.to_dict(),
                        "metrics": m,
                    }
                    for (item, res), (_, m) in zip(scored, pairs)
                ],
            })

        run_summary = scoring.summarize_run(rows)
        _print_run(run_summary, rounds_out)
        print(f"\n(done in {time.time() - started:.0f}s — predictions, not grades: "
              "see MIID/benchmark/scoring.py for which rules are published and which are estimated)")

        if args.json_path:
            Path(args.json_path).write_text(json.dumps(
                {"summary": run_summary, "rounds": rounds_out, "thresholds": scoring.T},
                indent=2, default=_json_default,
            ))
            print(f"JSON written to {args.json_path}")
        if args.html_path:
            from MIID.benchmark.report import write_html

            write_html(Path(args.html_path), rounds_out, run_summary)
            print(f"HTML report written to {args.html_path}")
    finally:
        analyzer.close()
    return 0
