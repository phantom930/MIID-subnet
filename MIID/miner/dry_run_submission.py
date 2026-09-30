r"""Offline dry run of the miner's submission pipeline.

Builds an ImageRequest locally and pushes it through the exact code a
validator query runs — generate variations, AdaFace identity check, drand
timelock encryption, upload — then prints the resulting S3Submission list.
No validator, no registered hotkey, no served axon required.

Use it to answer "can this box actually produce a submission, and if not,
which stage breaks?" before pointing a real validator at it.

Usage:
    # Simplest: local storage, throwaway key, the standard 5-variation challenge
    python -m MIID.miner.dry_run_submission --image /path/to/face.png

    # Use your real hotkey so signatures and the S3 key path match production
    python -m MIID.miner.dry_run_submission --image face.png \
        --wallet-name my_wallet --wallet-hotkey my_hotkey

    # Keep the generated PNGs to eyeball them, and save the submission JSON
    python -m MIID.miner.dry_run_submission --image face.png \
        --save-images ./dryrun_images --output ./dryrun_submissions.json

    # Exercise one specific variation instead of the full challenge set
    python -m MIID.miner.dry_run_submission --image face.png \
        --variations "pose_edit:far,lighting_edit+expression_edit:medium"

By default nothing is uploaded to the shared bucket: encrypted output is
written under MIID_LOCAL_STORAGE (default /tmp/miid_submissions). Pass --s3
to exercise the real HTTP PUT, which requires --wallet-name/--wallet-hotkey
so the upload lands under your own hotkey prefix.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from MIID.protocol import ImageRequest, VariationRequest
from MIID.miner.request_archive import (
    archive_enabled,
    archive_images_enabled,
    media_dir,
    new_record,
    record_outcome,
    resolve_archive_dir,
    save_request_media,
    write_record,
)


# Searched in order when --image is omitted. seeds/ is populated by a running
# miner from the validator's image-of-the-day, so an operator who has been
# online already has a usable face here.
_IMAGE_SEARCH_DIRS = (
    Path(__file__).parent / "real_image_miner_guide" / "seeds",
    Path(__file__).parent.parent / "validator" / "fixed_image",
    Path(__file__).parent.parent / "validator" / "base_images",
)
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _find_base_image() -> Optional[Path]:
    """Locate a usable base face image when the caller did not name one."""
    for directory in _IMAGE_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        candidates = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if candidates:
            return candidates[0]
    return None


def _load_base_image(path: Path) -> Tuple[str, str]:
    """Return (filename, base64 payload) exactly as a validator would send it."""
    raw = path.read_bytes()
    return path.name, base64.b64encode(raw).decode("utf-8")


def _parse_variations(spec: str) -> List[VariationRequest]:
    """Parse a "type:intensity,type:intensity" spec into VariationRequests."""
    requests: List[VariationRequest] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            var_type, intensity = chunk.rsplit(":", 1)
        else:
            var_type, intensity = chunk, "medium"
        var_type = var_type.strip()
        intensity = intensity.strip() or "medium"
        if not var_type:
            continue
        requests.append(VariationRequest(
            type=var_type,
            intensity=intensity,
            description=f"Dry-run request for {var_type}",
            detail=f"{var_type} at {intensity} intensity.",
        ))
    if not requests:
        raise ValueError(f"No usable variations parsed from {spec!r}")
    return requests


def _standard_variations() -> List[VariationRequest]:
    """The same 5-variation challenge set a validator builds each round."""
    from MIID.validator.image_variations import (
        build_standard_challenge_variations,
        IMAGE_VARIATION_REQUIREMENTS,
    )

    return [
        VariationRequest(
            type=v["type"],
            intensity=v["intensity"],
            description=v["description"],
            # forward.py appends the shared requirements to every detail; keep
            # the prompt identical so the dry run generates what production does.
            detail=f"{v['detail']}. {IMAGE_VARIATION_REQUIREMENTS}",
        )
        for v in build_standard_challenge_variations()
    ]


def _resolve_hotkey(args: argparse.Namespace) -> Any:
    """Return the keypair used to sign this dry run's submissions."""
    if args.wallet_name and args.wallet_hotkey:
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        bt.logging.info(
            f"Signing with wallet hotkey {wallet.hotkey.ss58_address} "
            f"({args.wallet_name}/{args.wallet_hotkey})"
        )
        return wallet.hotkey

    from bittensor_wallet import Keypair

    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    bt.logging.info(
        f"No wallet given — signing with a throwaway keypair "
        f"({keypair.ss58_address}). Signatures will not verify against the "
        f"metagraph; pass --wallet-name/--wallet-hotkey for a faithful run."
    )
    return keypair


def _resolve_target_round(reveal_delay: int, want_encryption: bool) -> Tuple[int, int, bool]:
    """Compute the drand target round, degrading to no-encryption when offline.

    Returns (target_round, reveal_timestamp, encryption_possible).
    """
    if not want_encryption:
        return 0, int(time.time() + reveal_delay), False

    from MIID.miner.drand_encrypt import is_timelock_available

    if not is_timelock_available():
        bt.logging.warning(
            "bittensor.timelock is unavailable (needs bittensor>=9, which ships "
            "bittensor_drand). Continuing without encryption — a real submission "
            "must be timelocked, so fix this before mining for real."
        )
        return 0, int(time.time() + reveal_delay), False

    try:
        from MIID.validator.drand_utils import calculate_target_round

        target_round, reveal_timestamp = calculate_target_round(reveal_delay)
        bt.logging.info(
            f"Drand target round {target_round} "
            f"(reveal in {reveal_delay}s / {reveal_delay / 60:.0f} min)"
        )
        return target_round, reveal_timestamp, True
    except Exception as e:
        bt.logging.warning(
            f"Could not reach the drand API ({e}). Continuing without timelock "
            f"encryption — this exercises the rest of the pipeline, but a real "
            f"submission must be encrypted."
        )
        return 0, int(time.time() + reveal_delay), False


def _configure_storage(use_s3: bool, storage_dir: Optional[str]) -> str:
    """Point the uploader at local storage unless the caller opted into S3.

    s3_upload reads MIID_USE_S3 at import time, so flip the module attribute
    rather than the environment variable.
    """
    from MIID.miner import s3_upload

    if storage_dir:
        s3_upload.LOCAL_STORAGE_DIR = Path(storage_dir)

    if use_s3:
        if not s3_upload.USE_S3:
            bt.logging.warning(
                "--s3 was requested but MIID_USE_S3 disables uploads; "
                "falling back to local storage."
            )
        else:
            bt.logging.info(f"Uploading to the live bucket: {s3_upload.S3_BUCKET_NAME}")
            return f"s3://{s3_upload.S3_BUCKET_NAME}"
    else:
        s3_upload.USE_S3 = False

    return str(s3_upload.LOCAL_STORAGE_DIR)


def _print_report(
    report: List[Dict[str, Any]],
    submissions: List[Any],
    requested: int,
    destination: str,
) -> None:
    """Print a per-variation outcome table and a one-line verdict."""
    print("\n" + "=" * 78)
    print("DRY RUN RESULT")
    print("=" * 78)

    if not report:
        print("  No variations were processed.")
    for entry in report:
        status = entry.get("status", "?")
        marker = {"submitted": "OK  ", "dropped": "DROP", "failed": "FAIL"}.get(status, "??  ")
        line = f"  [{marker}] {entry.get('variation_type', '?')}"
        if status == "submitted":
            similarity = entry.get("identity_similarity")
            identity = "identity n/a" if similarity is None else f"identity {similarity:.3f}"
            line += (
                f"  {entry.get('bytes', 0)} bytes"
                f"  {'encrypted' if entry.get('encrypted') else 'RAW (unencrypted)'}"
                f"  {identity}"
                f"  kept attempt {entry.get('winning_attempt', 1)}"
                f"/{entry.get('attempts', 1)}"
                f"\n           {entry.get('s3_key', '')}"
            )
        else:
            line += f"  — {entry.get('reason', 'unknown')}"
        print(line)

    print("-" * 78)
    print(f"  Requested : {requested}")
    print(f"  Submitted : {len(submissions)}")
    print(f"  Written to: {destination}")

    if submissions:
        print("\n  Pipeline works end-to-end. A validator query would return "
              f"{len(submissions)} submission(s).")
    else:
        print("\n  No submissions produced — the miner would return an empty list "
              "and score 0. See the reasons above.")
    print("=" * 78 + "\n")


def _submission_to_dict(submission: Any) -> Dict[str, Any]:
    """Serialize an S3Submission across pydantic versions."""
    if hasattr(submission, "model_dump"):
        return submission.model_dump()
    if hasattr(submission, "dict"):
        return submission.dict()
    return dict(submission)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the miner submission pipeline offline against a local image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        help="Base face image. Defaults to the first image found in the miner "
             "seeds/ folder, then the validator fixed_image/ and base_images/ folders.",
    )
    parser.add_argument(
        "--variations",
        help='Comma-separated "type:intensity" pairs, e.g. '
             '"pose_edit:far,lighting_edit+expression_edit:medium". '
             "Defaults to the standard 5-variation challenge set.",
    )
    parser.add_argument(
        "--wallet-name", "--wallet.name", dest="wallet_name",
        help="Wallet whose hotkey signs the submissions (recommended).",
    )
    parser.add_argument(
        "--wallet-hotkey", "--wallet.hotkey", dest="wallet_hotkey",
        help="Hotkey within --wallet-name.",
    )
    parser.add_argument(
        "--challenge-id",
        help="Challenge id to sign against. Defaults to dryrun_<timestamp>.",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=None,
        help="AdaFace similarity a variation is regenerated to reach "
             "(default: the miner's MIID_IDENTITY_TARGET, 0.6). The best "
             "attempt is submitted either way.",
    )
    parser.add_argument(
        "--skip-identity-check", action="store_true",
        help="Accept the first generation of each variation without retrying. "
             "Diagnostic only — it shows what the model produced before any "
             "identity-driven regeneration.",
    )
    parser.add_argument(
        "--no-encrypt", action="store_true",
        help="Skip drand timelock encryption (sandbox only; real submissions must be encrypted).",
    )
    parser.add_argument(
        "--reveal-delay", type=int, default=None,
        help="Seconds until drand reveal (default: the validator's 2400).",
    )
    parser.add_argument(
        "--s3", action="store_true",
        help="Actually upload to the shared bucket. Requires a real wallet.",
    )
    parser.add_argument(
        "--storage-dir",
        help="Local storage directory for encrypted output (default: MIID_LOCAL_STORAGE).",
    )
    parser.add_argument(
        "--save-images",
        help="Directory to write the unencrypted generated variations into.",
    )
    parser.add_argument(
        "--output",
        help="Write the S3Submission list to this JSON file.",
    )
    parser.add_argument(
        "--archive-dir",
        help="Directory to store the request/submission record in, used as given "
             "(default: MIID_REQUEST_ARCHIVE, else <repo>/miner_requests).",
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Do not store a request/submission record for this run.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable bittensor debug logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.debug:
        bt.logging.set_debug(True)

    if args.s3 and not (args.wallet_name and args.wallet_hotkey):
        print(
            "error: --s3 uploads to the shared bucket, so it needs a real hotkey. "
            "Pass --wallet-name and --wallet-hotkey, or drop --s3 to write locally.",
            file=sys.stderr,
        )
        return 2

    # --- base image -------------------------------------------------------
    if args.image:
        image_path = Path(args.image).expanduser()
    else:
        found = _find_base_image()
        if found is None:
            print(
                "error: no base image given and none found on disk. "
                "Pass --image /path/to/face.png",
                file=sys.stderr,
            )
            return 2
        image_path = found
        print(f"No --image given; using {image_path}")

    if not image_path.is_file():
        print(f"error: {image_path} is not a file", file=sys.stderr)
        return 2

    # Import late so a missing image stack reports cleanly instead of at import.
    try:
        from MIID.miner.submission_builder import (
            build_image_submissions,
            DEFAULT_MIN_SIMILARITY,
        )
    except ImportError as e:
        print(
            f"error: the image-generation stack is not installed ({e}).\n"
            f"       pip install -r requirements-miner.txt",
            file=sys.stderr,
        )
        return 2

    image_filename, base64_image = _load_base_image(image_path)

    # --- challenge --------------------------------------------------------
    try:
        variation_requests = (
            _parse_variations(args.variations) if args.variations else _standard_variations()
        )
    except Exception as e:
        print(f"error: could not build the variation set: {e}", file=sys.stderr)
        return 2

    from MIID.validator.drand_utils import REVEAL_DELAY_SECONDS

    reveal_delay = args.reveal_delay if args.reveal_delay is not None else REVEAL_DELAY_SECONDS
    target_round, reveal_timestamp, encryption_possible = _resolve_target_round(
        reveal_delay, want_encryption=not args.no_encrypt
    )

    challenge_id = args.challenge_id or f"dryrun_{int(time.time())}"

    image_request = ImageRequest(
        base_image=base64_image,
        image_filename=image_filename,
        variation_requests=variation_requests,
        target_drand_round=target_round,
        reveal_timestamp=reveal_timestamp,
        challenge_id=challenge_id,
    )

    hotkey = _resolve_hotkey(args)
    destination = _configure_storage(use_s3=args.s3, storage_dir=args.storage_dir)

    min_similarity = (
        0.0 if args.skip_identity_check
        else (args.min_similarity if args.min_similarity is not None else DEFAULT_MIN_SIMILARITY)
    )

    print(
        f"\nDry run {challenge_id}\n"
        f"  image      : {image_path}\n"
        f"  variations : {', '.join(f'{v.type}({v.intensity})' for v in variation_requests)}\n"
        f"  encryption : {'drand timelock' if encryption_possible else 'OFF (raw bytes)'}\n"
        f"  identity   : {'no retries' if args.skip_identity_check else f'retry until AdaFace >= {min_similarity}'}\n"
        f"  destination: {destination}\n"
    )

    # --- archive this approach like the live miner does --------------------
    archiving = archive_enabled() and not args.no_archive
    archive_root = resolve_archive_dir(args.archive_dir)
    record = None
    save_images_dir = args.save_images
    if archiving:
        record = new_record(
            "dry_run",
            image_request=image_request,
            miner_hotkey=getattr(hotkey, "ss58_address", None),
        )
        if archive_images_enabled():
            save_request_media(record, image_request, archive_root)
            if not save_images_dir:
                target = media_dir(record, archive_root)
                save_images_dir = str(target) if target else None

    # --- run the real pipeline -------------------------------------------
    report: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    submissions = build_image_submissions(
        image_request,
        hotkey,
        min_similarity=min_similarity,
        encrypt=encryption_possible,
        save_images_dir=save_images_dir,
        report=report,
        meta=meta,
    )

    if record is not None:
        record_outcome(
            record,
            outcome="submitted" if submissions else "empty",
            submissions=submissions,
            report=report,
            meta=meta,
        )
        archived = write_record(record, archive_root)
        if archived:
            print(f"Archived this run to {archived}")

    _print_report(report, submissions, len(variation_requests), destination)

    payload = [_submission_to_dict(s) for s in submissions]
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(payload)} submission(s) to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    return 0 if submissions else 1


if __name__ == "__main__":
    sys.exit(main())
