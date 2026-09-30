# MIID/miner/submission_builder.py
#
# Phase 4: turns a validator ImageRequest into signed S3Submission objects.
#
# This is the exact path neurons/miner.py runs when a validator queries it.
# It lives here, rather than on the Miner neuron, so the same code can be
# exercised offline by `python -m MIID.miner.dry_run_submission` without a
# registered hotkey, a served axon, or a live validator.

import gc
import io
import os
import time
from typing import Any, Dict, List, Optional

import bittensor as bt
from PIL import Image

from MIID.protocol import S3Submission
from MIID.miner.image_generator import (
    IDENTITY_TARGET,
    decode_base_image,
    generate_variations,
    validate_face_variation,
)
from MIID.miner.drand_encrypt import encrypt_image_for_drand, is_timelock_available
from MIID.miner.s3_upload import upload_to_s3


# AdaFace cosine-similarity a variation should reach. It is a *retry target*,
# not a drop floor: generate_variations() re-rolls a slot that misses it with
# the edit held back, and the best attempt is submitted either way.
#
# Submitting a weak variation beats dropping it. The validator averages over
# the requested types and scores a missing one as validation_score 0 AND
# identity_preservation 0 (MIID/validator/reward.py), so a dropped slot is
# strictly worse than any image that arrives — including on the identity
# threshold that decides whether the miner is ranked at all. Worse still, the
# grading API samples one slot per round: drop the slot it picked and the whole
# round scores zero.
#
# Set MIID_DROP_BELOW_SIMILARITY=1 to restore the old drop-on-miss behaviour.
DEFAULT_MIN_SIMILARITY = IDENTITY_TARGET


def _drop_below_similarity() -> bool:
    """Whether a variation under the identity target is dropped, not submitted."""
    return os.environ.get("MIID_DROP_BELOW_SIMILARITY", "0").strip().lower() in (
        "1", "true", "yes",
    )

# Extensions stripped from image_filename to get the S3 seed-name component.
_SEED_NAME_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Container signatures for the short screen-replay videos miners may submit.
_MP4_EXTENSIONS = (".mp4", ".mov", ".m4v")
_WEBM_MAGIC = b"\x1a\x45\xdf\xa3"


def free_gpu_memory(stage: str = "") -> None:
    """Release inter-request GPU memory and log resident VRAM."""
    try:
        gc.collect()
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            try:
                _torch.cuda.ipc_collect()
            except Exception:
                pass
            free_b, total_b = _torch.cuda.mem_get_info(0)
            reserved_b = _torch.cuda.memory_reserved(0)
            allocated_b = _torch.cuda.memory_allocated(0)
            gib = 1024 ** 3
            bt.logging.info(
                f"GPU mem [{stage}]: "
                f"free={free_b / gib:.2f} GiB / total={total_b / gib:.2f} GiB, "
                f"torch_reserved={reserved_b / gib:.2f} GiB, "
                f"torch_allocated={allocated_b / gib:.2f} GiB"
            )
    except Exception as _e:
        bt.logging.debug(f"free_gpu_memory({stage}) failed: {_e}")


def is_valid_image_bytes(image_bytes: bytes) -> bool:
    """Validate whether raw bytes represent a valid image."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except Exception:
        return False


def is_valid_video_bytes(video_bytes: bytes, suffix: str = "") -> bool:
    """Lightweight check that bytes look like a short screen-replay video."""
    if not video_bytes or len(video_bytes) < 32:
        return False
    head = video_bytes[:64]
    ext = (suffix or "").lower()
    if ext in _MP4_EXTENSIONS:
        return b"ftyp" in head
    if ext == ".webm":
        return head.startswith(_WEBM_MAGIC)
    return b"ftyp" in head or head.startswith(_WEBM_MAGIC)


def seed_name_from_filename(image_filename: str) -> str:
    """Strip a known image extension to get the S3 seed-name component."""
    seed_image_name = image_filename or "seed"
    for ext in _SEED_NAME_EXTENSIONS:
        if seed_image_name.endswith(ext):
            return seed_image_name[: -len(ext)]
    return seed_image_name


def build_path_signature(hotkey: Any, challenge_id: str) -> str:
    """Sign the challenge/hotkey pair to namespace this miner's S3 prefix.

    Only the holder of the hotkey's private key can produce this value, which
    is what stops one miner writing into another miner's submission path.
    """
    path_message = f"{challenge_id}:{hotkey.ss58_address}"
    return hotkey.sign(path_message.encode()).hex()[:16]


def sign_media_hash(hotkey: Any, challenge_id: str, image_hash: str) -> str:
    """Sign one media hash, proving this miner produced that exact file."""
    message = f"challenge:{challenge_id}:hash:{image_hash}"
    return hotkey.sign(message.encode()).hex()


def _record(
    report: Optional[List[Dict[str, Any]]],
    variation_type: str,
    status: str,
    **fields: Any,
) -> None:
    """Append a per-variation outcome when the caller asked for a report."""
    if report is None:
        return
    entry: Dict[str, Any] = {"variation_type": variation_type, "status": status}
    entry.update(fields)
    report.append(entry)


def build_image_submissions(
    image_request: Any,
    hotkey: Any,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    encrypt: Optional[bool] = None,
    save_images_dir: Optional[str] = None,
    report: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> List[S3Submission]:
    """Generate, validate, encrypt and upload every requested variation.

    Args:
        image_request: The validator's ImageRequest (base image + requests).
        hotkey: Keypair used to sign submissions (``wallet.hotkey`` in the
            live miner).
        min_similarity: AdaFace similarity a variation is regenerated to reach.
            The best attempt is submitted whether or not it gets there, unless
            MIID_DROP_BELOW_SIMILARITY is set — see DEFAULT_MIN_SIMILARITY.
        encrypt: ``None`` (default) encrypts whenever drand timelock is
            available — this is what the live miner does. ``False`` forces
            raw bytes and is sandbox/dry-run only: real submissions must be
            timelocked or the validator can read them before reveal.
        save_images_dir: When set, the unencrypted variation is also written
            here so it can be inspected. Used by the dry run, and by the live
            miner when MIID_ARCHIVE_IMAGES is on.
        report: When a list is passed, one dict per variation is appended
            describing what happened to it (submitted, or why it was dropped).
        meta: When a dict is passed, run-level facts are written into it —
            which model was picked, the identity floor, whether encryption was
            used, and how many variations were requested vs submitted. The
            request archive stores this so a past round can be explained.

    Returns:
        List of S3Submission objects, one per variation that made it through.
    """
    if meta is not None:
        meta.update({
            "model_key": None,
            "model_id": None,
            "min_similarity": min_similarity,
            "encrypted": None,
            "requested": len(getattr(image_request, "variation_requests", None) or []),
            "generated": 0,
            "submitted": 0,
        })

    if image_request is None:
        return []

    free_gpu_memory("before_request")

    base_image = None
    variations = None
    model_key = None
    model_id = None
    try:
        bt.logging.info(f"Phase 4: Decoding base image: {image_request.image_filename}")
        base_image = decode_base_image(image_request.base_image)

        seed_image_name = seed_name_from_filename(image_request.image_filename)

        bt.logging.info(
            f"Phase 4: Generating {image_request.requested_variations} variations "
            f"(from validator: {[f'{v.type}({v.intensity})' for v in image_request.variation_requests]})"
        )
        generation_started = time.monotonic()
        variations = generate_variations(
            base_image,
            image_request.variation_requests,
            identity_target=min_similarity,
        )
        generation_seconds = time.monotonic() - generation_started

        # Which approach produced this batch — identical across the call, but
        # read off the results so it reflects what actually ran, including the
        # fallback that kicks in when the chosen model fails to load.
        model_key = variations[0].get("model_key") if variations else None
        model_id = variations[0].get("model_id") if variations else None
        if meta is not None:
            meta["model_key"] = model_key
            meta["model_id"] = model_id
            meta["generated"] = len(variations)
            # Archived so a slow round can be attributed: generation dominates
            # the response, and the rest (encrypt + upload) is the remainder
            # against the record's duration_seconds.
            meta["generation_seconds"] = round(generation_seconds, 1)

        s3_submissions: List[S3Submission] = []
        target_round = image_request.target_drand_round
        challenge_id = image_request.challenge_id or "sandbox_test"

        # Generate path_signature once per challenge (prevents path hijacking)
        path_signature = build_path_signature(hotkey, challenge_id)
        bt.logging.debug(f"Phase 4: Generated path_signature: {path_signature}")

        if save_images_dir:
            os.makedirs(save_images_dir, exist_ok=True)

        should_encrypt = is_timelock_available() if encrypt is None else bool(encrypt)
        if meta is not None:
            meta["encrypted"] = should_encrypt

        for index, var in enumerate(variations):
            variation_type = var["variation_type"]
            try:
                if not is_valid_image_bytes(var["image_bytes"]):
                    bt.logging.warning(
                        f"Phase 4: Skipping invalid/corrupt image for {variation_type}"
                    )
                    _record(report, variation_type, "dropped", model=model_key,
                            reason="invalid or corrupt image bytes")
                    continue

                if save_images_dir:
                    safe_name = variation_type.replace("+", "_")
                    local_copy = os.path.join(
                        save_images_dir, f"{index:02d}_{safe_name}.png"
                    )
                    with open(local_copy, "wb") as f:
                        f.write(var["image_bytes"])
                    bt.logging.debug(f"Phase 4: Wrote {local_copy}")

                similarity = var.get("identity_similarity")
                meets_identity = validate_face_variation(
                    var, base_image, min_similarity=min_similarity
                )
                if not meets_identity:
                    shown = "no face detected" if similarity is None else f"{similarity:.3f}"
                    if _drop_below_similarity():
                        bt.logging.warning(
                            f"Phase 4: Dropping {variation_type} — identity "
                            f"{shown} < {min_similarity} "
                            f"(MIID_DROP_BELOW_SIMILARITY is set)"
                        )
                        _record(report, variation_type, "dropped", model=model_key,
                                identity_similarity=similarity,
                                attempts=var.get("attempts"),
                                winning_attempt=var.get("winning_attempt"),
                                reason=f"AdaFace similarity below {min_similarity}")
                        continue
                    bt.logging.warning(
                        f"Phase 4: Submitting {variation_type} below the identity "
                        f"target ({shown} < {min_similarity}) — best of "
                        f"{var.get('attempts', 1)} attempt(s); a missing slot "
                        f"would score lower still"
                    )

                signature = sign_media_hash(hotkey, challenge_id, var["image_hash"])

                if should_encrypt:
                    encrypted_data = encrypt_image_for_drand(var["image_bytes"], target_round)
                    if encrypted_data is None:
                        bt.logging.warning(f"Phase 4: Encryption failed for {variation_type}")
                        _record(report, variation_type, "dropped", model=model_key,
                                reason="drand timelock encryption failed")
                        continue
                else:
                    bt.logging.warning("Phase 4: Timelock not available, using raw bytes (SANDBOX ONLY)")
                    encrypted_data = var["image_bytes"]

                s3_key = upload_to_s3(
                    encrypted_data=encrypted_data,
                    miner_hotkey=hotkey.ss58_address,
                    signature=signature,
                    image_hash=var["image_hash"],
                    target_round=target_round,
                    challenge_id=challenge_id,
                    variation_type=variation_type,
                    path_signature=path_signature,
                    seed_image_name=seed_image_name,
                )

                if s3_key:
                    s3_submissions.append(S3Submission(
                        s3_key=s3_key,
                        image_hash=var["image_hash"],
                        signature=signature,
                        variation_type=variation_type,
                        path_signature=path_signature,
                    ))
                    bt.logging.debug(f"Phase 4: Created submission for {variation_type}")
                    _record(report, variation_type, "submitted", model=model_key,
                            s3_key=s3_key,
                            image_hash=var["image_hash"],
                            encrypted=should_encrypt,
                            identity_similarity=similarity,
                            attempts=var.get("attempts"),
                            winning_attempt=var.get("winning_attempt"),
                            bytes=len(encrypted_data))
                else:
                    bt.logging.warning(
                        f"Phase 4: Upload returned no key for {variation_type}"
                    )
                    _record(report, variation_type, "dropped", model=model_key,
                            reason="upload failed (S3 and local storage)")

            except Exception as e:
                bt.logging.error(f"Phase 4: Error processing variation {variation_type}: {e}")
                _record(report, variation_type, "dropped", model=model_key,
                        reason=f"{type(e).__name__}: {e}")
                continue

        bt.logging.info(f"Phase 4: Successfully created {len(s3_submissions)} S3 submissions")
        if meta is not None:
            meta["submitted"] = len(s3_submissions)
        return s3_submissions

    except Exception as e:
        bt.logging.error(f"Phase 4: Error in build_image_submissions: {e}")
        _record(report, "*", "failed", model=model_key,
                reason=f"{type(e).__name__}: {e}")
        if meta is not None:
            meta["error"] = f"{type(e).__name__}: {e}"
        return []
    finally:
        base_image = None
        variations = None
        free_gpu_memory("after_request")
