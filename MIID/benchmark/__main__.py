r"""Predict the grading API's validation_score for submissions already made.

Reads request-archive records (written with MIID_ARCHIVE_IMAGES=1), re-measures
each submitted image the way the score sheet describes, and prints the score
the grader would most likely give, why, and what that means for KAV reward and
reputation (Δrep). See MIID/benchmark/scoring.py for every rule and its source.

Usage:
    # Every archived round with images
    python -m MIID.benchmark

    # The 10 most recent rounds, with an HTML report to eyeball the images
    python -m MIID.benchmark --last 10 --html bench.html

    # One image outside the archive (e.g. trying a new model or prompt)
    python -m MIID.benchmark --base face.png --image out.png \
        --type lighting_edit+pose_edit --intensity medium+far

Runs on CPU by default so it never competes with a live miner for VRAM. The
first run downloads the MediaPipe task files and CLIP; measurements are then
cached per record (benchmark.json) and only scoring is redone on later runs.

It needs MediaPipe, which miner_env does not ship; scripts/miner/benchmark.sh
sets up a separate bench_env layered on top of miner_env and runs this.
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python -m MIID.benchmark",
        description="Predict validation scores for archived miner submissions.",
    )
    p.add_argument("paths", nargs="*",
                   help="archive root, date dir or record dir (default: the request archive)")
    p.add_argument("--last", type=int, default=0, help="only the N most recent records")
    p.add_argument("--include-dropped", action="store_true",
                   help="also score variations the miner generated but did not submit")
    p.add_argument("--json", dest="json_path", help="write every result to this JSON file")
    p.add_argument("--html", dest="html_path", help="write an HTML report with thumbnails")
    p.add_argument("--force", action="store_true", help="ignore cached measurements")
    p.add_argument("--no-clip", action="store_true",
                   help="skip CLIP (accessory, environment, gender fallback)")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda — cuda shares the miner's GPU")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for torch (default 8)")
    p.add_argument("-v", "--verbose", action="store_true", help="show detections for every variation")

    adhoc = p.add_argument_group("single image")
    adhoc.add_argument("--base", help="base / seed face image")
    adhoc.add_argument("--image", help="variation image to score")
    adhoc.add_argument("--type", dest="var_type", help="e.g. lighting_edit+pose_edit or background_in")
    adhoc.add_argument("--intensity", default="medium", help="e.g. medium or light+far")
    adhoc.add_argument("--detail", default="",
                       help="request detail text; include 'Additionally, include: <accessory>' "
                            "for a background accessory")
    return p.parse_args(argv)


def _isolate_device(args) -> None:
    """Pin every model to the requested device before torch is imported.

    ada_face_compare picks its MTCNN device at import time from
    ADA_FACE_DEVICE / FLUX_DEVICE, and a miner shell usually exports
    FLUX_DEVICE=cuda — which would open a CUDA context on the GPU the live
    miner is filling.
    """
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["ADA_FACE_DEVICE"] = "cpu"
        os.environ["FLUX_DEVICE"] = "cpu"
    else:
        os.environ["ADA_FACE_DEVICE"] = args.device
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _isolate_device(args)

    from MIID.benchmark.runner import run

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
