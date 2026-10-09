# MIID/miner/speech_brain_compare.py
#
# Speaker-identity check for voice generation.
#
# Same role as AdaFace for faces: gate a generated voice clone here before
# encrypt/upload. The validator scores voice identity with SpeechBrain
# speaker verification (speechbrain/spkrec-ecapa-voxceleb) and counts a clone
# as identity-preserving above 0.6 (P5 C1).
#
# The ECAPA model runs inside the voice worker (voice_env/, see
# voice_generator.py), not in miner_env.

import os
import tempfile
from typing import Optional

import bittensor as bt

from MIID.miner.voice_generator import voice_env_available, worker_call


SPEECHBRAIN_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MIN_SIMILARITY = 0.6


def voice_similarity(base_wav: bytes, generated_wav: bytes) -> Optional[float]:
    """ECAPA cosine similarity of two WAVs, or None if it could not be measured."""
    with tempfile.TemporaryDirectory(prefix="miid_voice_sim_") as tmp:
        ref = os.path.join(tmp, "reference.wav")
        gen = os.path.join(tmp, "generated.wav")
        with open(ref, "wb") as f:
            f.write(base_wav)
        with open(gen, "wb") as f:
            f.write(generated_wav)
        reply = worker_call("similarity", ref_wav=ref, gen_wav=gen, timeout=120)
    if not reply or not reply.get("ok"):
        return None
    return float(reply["similarity"])


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
        True if identity is considered preserved. When voice_env is missing
        the check cannot run and the submission is let through, as before.
    """
    if not base_wav or not generated_wav:
        bt.logging.warning("Voice identity: empty wav bytes")
        return False
    if not voice_env_available():
        bt.logging.debug(
            "Voice identity: voice_env not set up; accepting submission unchecked"
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
