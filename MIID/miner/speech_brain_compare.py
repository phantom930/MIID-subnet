# MIID/miner/speech_brain_compare.py
#
# Speaker-identity check for voice generation.
#
# Same role as AdaFace for faces: gate a generated voice clone here before
# encrypt/upload. The validator scores voice identity with SpeechBrain
# speaker verification (speechbrain/spkrec-ecapa-voxceleb) and counts a clone
# as identity-preserving above 0.6 (P5 C1).
#
# The ECAPA model runs in this process; voice_generator loads it alongside
# Chatterbox and parks both in CPU memory between rounds.

from typing import Optional

import bittensor as bt

from MIID.miner.voice_generator import voice_available, voice_similarity  # noqa: F401


DEFAULT_MIN_SIMILARITY = 0.6


def validate_voice_identity(
    base_wav: bytes,
    generated_wav: bytes,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    model: Optional[object] = None,
) -> bool:
    """Check that generated audio preserves the reference speaker identity.

    Args:
        base_wav: Reference speaker WAV bytes.
        generated_wav: Miner-generated WAV bytes.
        min_similarity: Minimum cosine similarity (default 0.6).
        model: Unused; kept for interface compatibility.

    Returns:
        True if identity is considered preserved. When the voice packages are
        not installed the check cannot run and the submission is let through,
        as before.
    """
    if not base_wav or not generated_wav:
        bt.logging.warning("Voice identity: empty wav bytes")
        return False
    if not voice_available():
        bt.logging.debug(
            "Voice identity: voice packages not installed; accepting submission unchecked"
        )
        return True

    similarity = voice_similarity(base_wav, generated_wav)
    if similarity is None:
        bt.logging.warning("Voice identity: similarity could not be measured")
        return False
    passed = similarity >= min_similarity
    bt.logging.info(
        f"Voice identity: ECAPA similarity {similarity:.3f} "
        f"({'pass' if passed else 'fail'}, threshold {min_similarity})"
    )
    return passed
