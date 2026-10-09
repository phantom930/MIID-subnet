#!/usr/bin/env bash
# Set up voice_env/ for the P5 voice-clone challenge.
#
#   bash scripts/miner/setup_voice.sh            # create (or update) voice_env
#   bash scripts/miner/setup_voice.sh --warmup   # also download the models now
#
# Voice cloning (Chatterbox Multilingual, en + es) and the SpeechBrain ECAPA
# identity check run in a separate worker process from this environment.
# Chatterbox pins an older torch/transformers stack than the FLUX image path
# needs, so installing it into miner_env would break image generation on a
# live miner. The miner finds this env at <repo>/voice_env (override with
# MIID_VOICE_ENV) and starts MIID/miner/voice_worker.py in it on demand.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VOICE_ENV="${MIID_VOICE_ENV:-$REPO/voice_env}"
PY_VERSION="${VOICE_PYTHON:-3.11}"
# setuptools<81: resemble-perth (Chatterbox's watermarker) imports pkg_resources,
# which newer setuptools and bare uv venvs do not provide.
PACKAGES=(chatterbox-tts speechbrain soundfile "setuptools<81")

if command -v uv >/dev/null 2>&1; then
    [[ -x "$VOICE_ENV/bin/python" ]] || uv venv --python "$PY_VERSION" "$VOICE_ENV"
    VIRTUAL_ENV="$VOICE_ENV" uv pip install "${PACKAGES[@]}"
else
    if [[ ! -x "$VOICE_ENV/bin/python" ]]; then
        "python$PY_VERSION" -m venv "$VOICE_ENV" || {
            echo "python$PY_VERSION not found — install it or uv (https://docs.astral.sh/uv/)." >&2
            exit 1
        }
    fi
    "$VOICE_ENV/bin/pip" install --upgrade pip
    "$VOICE_ENV/bin/pip" install "${PACKAGES[@]}"
fi

"$VOICE_ENV/bin/python" -W ignore -c "import chatterbox.mtl_tts, speechbrain, soundfile, perth; assert perth.PerthImplicitWatermarker, 'perth watermarker failed to import'; print('voice_env OK')"

if [[ "${1:-}" == "--warmup" ]]; then
    # Pull the weights into the HF cache now so the first voice request
    # does not spend its time budget downloading them.
    "$VOICE_ENV/bin/python" - <<'EOF'
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from speechbrain.inference.speaker import EncoderClassifier
import os
ChatterboxMultilingualTTS.from_pretrained(device="cpu")
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
                         "speechbrain", "spkrec-ecapa-voxceleb"),
)
print("voice models cached")
EOF
fi
