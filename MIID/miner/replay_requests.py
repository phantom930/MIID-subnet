# MIID/miner/replay_requests.py
#
# Replay archived validator requests through the current miner pipeline.
#
# Each archived record (miner_requests/<date>/<record>/record.json, written
# with MIID_ARCHIVE_IMAGES=1 so the base image is on disk) is regenerated with
# image_generator.generate_variations — model choice, prompts, AdaFace and
# head-turn retries exactly as live — and written as a new record directory
# that `bash scripts/miner/benchmark.sh <out>/<date>/*` can score. Use it to
# compare a change against what the miner actually submitted for the same
# requests (e.g. MIID_INFERENCE_STEPS=6 vs 8).
#
# Needs the GPU, so stop the live miner first:
#   python -m MIID.miner.replay_requests --out /tmp/replay \
#       miner_requests/2026-10-08/2026*  [--deadline-seconds 400] [--only-background]

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

from MIID.miner import image_generator
from MIID.miner.generate_variations import subject_gender_from_filename


def replay(record_dir: Path, out_root: Path, deadline_seconds=None, only_background=False) -> dict:
    dest = out_root / record_dir.parent.name / record_dir.name
    if (dest / "record.json").exists():
        print(f"skip {record_dir.name} (already replayed)", flush=True)
        return {}
    record = json.loads((record_dir / "record.json").read_text())
    request = record["request"]
    if only_background:
        request["variation_requests"] = [
            r for r in request["variation_requests"] if r["type"].startswith("background")
        ]
    base_path = next((record_dir / "media").glob("base_image__*"))
    base = Image.open(base_path).convert("RGB")
    (dest / "media").mkdir(parents=True, exist_ok=True)
    shutil.copy(base_path, dest / "media" / base_path.name)

    started = time.time()
    variations = image_generator.generate_variations(
        base,
        request["variation_requests"],
        subject_gender=subject_gender_from_filename(request.get("image_filename")),
        deadline=started + deadline_seconds if deadline_seconds else None,
    )
    seconds = time.time() - started

    report = []
    for index, var in enumerate(variations):
        name = var["variation_type"].replace("+", "_")
        (dest / "media" / f"{index:02d}_{name}.png").write_bytes(var["image_bytes"])
        report.append({
            "variation_type": var["variation_type"],
            "status": "submitted",
            "model": var["model_key"],
            "identity_similarity": var["identity_similarity"],
            "attempts": var["attempts"],
            "winning_attempt": var["winning_attempt"],
        })
    record["variation_report"] = report
    record["generation"] = {
        "model_key": variations[0]["model_key"] if variations else None,
        "replayed_from": str(record_dir),
        "generation_seconds": round(seconds, 1),
    }
    record["duration_seconds"] = round(seconds, 1)
    (dest / "record.json").write_text(json.dumps(record, indent=2, default=str))

    attempts = sum(r["attempts"] for r in report)
    print(
        f"{record_dir.name[:16]} {seconds:.0f}s, {attempts} attempts "
        f"(~{image_generator.cycle_estimate():.1f}s each): "
        + " ".join(
            f"{r['variation_type']}={r['identity_similarity'] or 0:.2f}/{r['attempts']}"
            for r in report
        ),
        flush=True,
    )
    return {"seconds": seconds, "attempts": attempts, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="archived record directories")
    parser.add_argument("--out", type=Path, required=True, help="output root")
    parser.add_argument("--deadline-seconds", type=float, default=None,
                        help="simulate a validator deadline this many seconds after start")
    parser.add_argument("--only-background", action="store_true",
                        help="replay only the background/accessory slots")
    args = parser.parse_args()

    results = [
        r for r in (
            replay(d, args.out, args.deadline_seconds, args.only_background)
            for d in args.records if (d / "record.json").exists()
        ) if r
    ]
    if not results:
        return 1
    identities = [
        r["identity_similarity"] for res in results for r in res["report"]
        if r["identity_similarity"] is not None
    ]
    print(
        f"\n{len(results)} records, {sum(r['attempts'] for r in results)} attempts, "
        f"{statistics.median(r['seconds'] for r in results):.0f}s median per record, "
        f"~{image_generator.cycle_estimate():.1f}s per attempt; identity mean "
        f"{statistics.mean(identities):.3f}, "
        f"{sum(1 for i in identities if i < image_generator.IDENTITY_TARGET)}/{len(identities)} "
        f"kept below {image_generator.IDENTITY_TARGET}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
