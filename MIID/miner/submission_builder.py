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
from typing import Any, Dict, List, Optional

import bittensor as bt
from PIL import Image

from MIID.protocol import S3Submission
from MIID.miner.image_generator import (
    decode_base_image,
    generate_variations,
    validate_face_variation,
)
from MIID.miner.drand_encrypt import encrypt_image_for_drand, is_timelock_available
from MIID.miner.s3_upload import upload_to_s3


# AdaFace cosine-similarity floor. Variations below it are dropped instead of
# submitted: the grading API scores identity preservation, so shipping a
# variation that lost the face is worse than shipping nothing for that slot.
DEFAULT_MIN_SIMILARITY = 0.4

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
) -> List[S3Submission]:
    """Generate, validate, encrypt and upload every requested variation.

    Args:
        image_request: The validator's ImageRequest (base image + requests).
        hotkey: Keypair used to sign submissions (``wallet.hotkey`` in the
            live miner).
        min_similarity: AdaFace floor below which a variation is dropped.
        encrypt: ``None`` (default) encrypts whenever drand timelock is
            available — this is what the live miner does. ``False`` forces
            raw bytes and is sandbox/dry-run only: real submissions must be
            timelocked or the validator can read them before reveal.
        save_images_dir: When set, the unencrypted variation is also written
            here so it can be inspected. Never used on the live path.
        report: When a list is passed, one dict per variation is appended
            describing what happened to it (submitted, or why it was dropped).

    Returns:
        List of S3Submission objects, one per variation that made it through.
    """
    if image_request is None:
        return []

    free_gpu_memory("before_request")

    base_image = None
    variations = None
    try:
        bt.logging.info(f"Phase 4: Decoding base image: {image_request.image_filename}")
        base_image = decode_base_image(image_request.base_image)

        seed_image_name = seed_name_from_filename(image_request.image_filename)

        bt.logging.info(
            f"Phase 4: Generating {image_request.requested_variations} variations "
            f"(from validator: {[f'{v.type}({v.intensity})' for v in image_request.variation_requests]})"
        )
        variations = generate_variations(
            base_image,
            image_request.variation_requests
        )

        s3_submissions: List[S3Submission] = []
        target_round = image_request.target_drand_round
        challenge_id = image_request.challenge_id or "sandbox_test"

        # Generate path_signature once per challenge (prevents path hijacking)
        path_signature = build_path_signature(hotkey, challenge_id)
        bt.logging.debug(f"Phase 4: Generated path_signature: {path_signature}")

        if save_images_dir:
            os.makedirs(save_images_dir, exist_ok=True)

        should_encrypt = is_timelock_available() if encrypt is None else bool(encrypt)

        for index, var in enumerate(variations):
            variation_type = var["variation_type"]
            try:
                if not is_valid_image_bytes(var["image_bytes"]):
                    bt.logging.warning(
                        f"Phase 4: Skipping invalid/corrupt image for {variation_type}"
                    )
                    _record(report, variation_type, "dropped",
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

                if not validate_face_variation(var, base_image, min_similarity=min_similarity):
                    bt.logging.warning(
                        f"Phase 4: Skipping {variation_type} — face identity not preserved"
                    )
                    _record(report, variation_type, "dropped",
                            reason=f"AdaFace similarity below {min_similarity}")
                    continue

                signature = sign_media_hash(hotkey, challenge_id, var["image_hash"])

                if should_encrypt:
                    encrypted_data = encrypt_image_for_drand(var["image_bytes"], target_round)
                    if encrypted_data is None:
                        bt.logging.warning(f"Phase 4: Encryption failed for {variation_type}")
                        _record(report, variation_type, "dropped",
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
                    _record(report, variation_type, "submitted",
                            s3_key=s3_key,
                            image_hash=var["image_hash"],
                            encrypted=should_encrypt,
                            bytes=len(encrypted_data))
                else:
                    bt.logging.warning(
                        f"Phase 4: Upload returned no key for {variation_type}"
                    )
                    _record(report, variation_type, "dropped",
                            reason="upload failed (S3 and local storage)")

            except Exception as e:
                bt.logging.error(f"Phase 4: Error processing variation {variation_type}: {e}")
                _record(report, variation_type, "dropped", reason=f"{type(e).__name__}: {e}")
                continue

        bt.logging.info(f"Phase 4: Successfully created {len(s3_submissions)} S3 submissions")
        return s3_submissions

    except Exception as e:
        bt.logging.error(f"Phase 4: Error in build_image_submissions: {e}")
        _record(report, "*", "failed", reason=f"{type(e).__name__}: {e}")
        return []
    finally:
        base_image = None
        variations = None
        free_gpu_memory("after_request")
