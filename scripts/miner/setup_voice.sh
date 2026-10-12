#!/usr/bin/env bash
# Install the voice-clone stack (P5 C1) into miner_env, next to the image stack.
#
#   bash scripts/miner/setup_voice.sh            # install (or update) the packages
#   bash scripts/miner/setup_voice.sh --warmup   # also download the model weights now
#
# Voice cloning (Chatterbox Multilingual, en + es) and the SpeechBrain ECAPA
# identity check run inside the miner process (MIID/miner/voice_generator.py),
# on the same GPU as the FLUX image path.
#
# chatterbox-tts publishes pins (torch==2.6, transformers==5.2, diffusers==0.29,
# numpy<2) that would downgrade and break the image stack, so it is installed
# WITHOUT its pins and the packages it actually needs at runtime come from
# requirements-voice.txt. `pip check` will keep listing those pins as unmet;
# that is expected. Verified on torch 2.14 / transformers 5.19 / diffusers 0.41.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The interpreter to install into: MIID_PYTHON, else the active virtualenv,
# else <repo>/miner_env.
PY="${MIID_PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PY="$VIRTUAL_ENV/bin/python"
    elif [[ -x "$REPO/miner_env/bin/python" ]]; then
        PY="$REPO/miner_env/bin/python"
    else
        echo "No miner_env found. Activate the miner's virtualenv first (or set MIID_PYTHON)." >&2
        exit 1
    fi
fi
echo "Installing the voice stack into: $PY"

if ! "$PY" -c "import torch, diffusers, transformers" >/dev/null 2>&1; then
    echo "Install the image stack first:  pip install -e . && pip install -r requirements-miner.txt" >&2
    exit 1
fi

"$PY" -m pip install -r "$REPO/requirements-voice.txt"
"$PY" -m pip install --no-deps "chatterbox-tts>=0.1.7"

"$PY" -W ignore -c "
import chatterbox.mtl_tts, speechbrain.inference.speaker, soundfile, perth
assert perth.PerthImplicitWatermarker, 'perth watermarker failed to import'
from diffusers import Flux2KleinPipeline  # the image stack must still import
print('voice stack OK')
"

if [[ "${1:-}" == "--warmup" ]]; then
    # Pull the weights (~3 GB) into the HF cache now. The miner also does this
    # in the background at startup (MIID_VOICE_PRELOAD), but doing it here
    # shows download problems before the miner is live.
    "$PY" -W ignore - <<'EOF'
import os
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from speechbrain.inference.speaker import EncoderClassifier
ChatterboxMultilingualTTS.from_pretrained(device="cpu")
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")),
                         "speechbrain", "spkrec-ecapa-voxceleb"),
)
print("voice models cached")
EOF
fi
