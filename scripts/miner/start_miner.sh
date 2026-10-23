#!/usr/bin/env bash
# Start the miner under pm2: one process, one hotkey, the whole GPU.
#
#   bash scripts/miner/start_miner.sh                  # hotkey_0 on axon port 8095
#   bash scripts/miner/start_miner.sh hotkey_0:8095    # <hotkey>:<axon port>
#   bash scripts/miner/start_miner.sh --dry-run        # print the pm2 command only
#
# The public port is taken from VAST_TCP_PORT_<axon port> (Vast.ai) when that
# variable exists, else the axon port itself. An existing pm2 process named
# miid_miner is deleted and started again — that is a restart of a live
# miner, so run it right after a "Request completed" line in its log.
#
# HF_TOKEN comes from the environment, else ${WORKSPACE}/.env, else the
# token file `huggingface-cli login` writes ($HF_HOME/token). Without it
# every image request fails with "Missing Hugging Face token", so the
# script refuses to start. CPU thread pools default to the whole cgroup
# quota (MIID_CPU_THREADS overrides).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${MINER_PYTHON:-$REPO/miner_env/bin/python}"
COLDKEY="${MIID_COLDKEY:-phantom_coldkey}"
NETUID="${MIID_NETUID:-54}"
NETWORK="${MIID_NETWORK:-finney}"
ENDPOINT="${MIID_CHAIN_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"
EXTERNAL_IP="${MIID_EXTERNAL_IP:-${PUBLIC_IPADDR:-}}"
NAME="${MIID_PM2_NAME:-miid_miner}"

DRY_RUN=0
SPEC="hotkey_0:8095"
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n 2,17p "${BASH_SOURCE[0]}"; exit 0 ;;
        *) SPEC="$a" ;;
    esac
done
hotkey="${SPEC%%:*}"
port="${SPEC##*:}"
[[ "$hotkey" != "$SPEC" && "$port" =~ ^[0-9]+$ ]] || { echo "bad argument '$SPEC' (want <hotkey>:<port>)" >&2; exit 1; }

# nvm's node/pm2 are only on PATH in login shells.
if ! command -v pm2 >/dev/null 2>&1 && [[ -f /opt/nvm/nvm.sh ]]; then
    # shellcheck disable=SC1091
    . /opt/nvm/nvm.sh
fi
command -v pm2 >/dev/null 2>&1 || { echo "pm2 not found on PATH" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "miner python not found at $PYTHON (set MINER_PYTHON)" >&2; exit 1; }
[[ -f "$HOME/.bittensor/wallets/$COLDKEY/hotkeys/$hotkey" ]] || { echo "no key file for $COLDKEY/$hotkey" >&2; exit 1; }

if [[ -z "${HF_TOKEN:-}" && -f "${WORKSPACE:-/workspace}/.env" ]]; then
    HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "${WORKSPACE:-/workspace}/.env" | tr -d '"' | head -1)"
fi
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set (export it, or put HF_TOKEN=... in ${WORKSPACE:-/workspace}/.env)" >&2; exit 1; }
[[ -n "$EXTERNAL_IP" ]] || { echo "External IP unknown: set MIID_EXTERNAL_IP (PUBLIC_IPADDR is empty)" >&2; exit 1; }

public_var="VAST_TCP_PORT_${port}"
public_port="${!public_var:-$port}"

cmd=("$PYTHON" neurons/miner.py
    --netuid "$NETUID" --subtensor.network "$NETWORK" --subtensor.chain_endpoint "$ENDPOINT"
    --wallet.name "$COLDKEY" --wallet.hotkey "$hotkey"
    --axon.port "$port" --axon.ip 0.0.0.0
    --axon.external_ip "$EXTERNAL_IP" --axon.external_port "$public_port"
    --logging.debug)

echo "== $NAME: $hotkey on :$port (public $EXTERNAL_IP:$public_port)"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "   pm2 start ... -- ${cmd[*]}"
    exit 0
fi
if pm2 describe "$NAME" >/dev/null 2>&1; then
    echo "   replacing existing pm2 process $NAME"
    pm2 delete "$NAME" >/dev/null
fi

# Memory: mapped offload keeps the FLUX weights in the page cache (~1.5 GB
# private instead of 15 GB, no copy-back churn); MALLOC_ARENA_MAX stops glibc
# giving each of the hundreds of threads its own 64 MB arena.
threads=()
if [[ -n "${MIID_CPU_THREADS:-}" ]]; then
    threads=(MIID_TORCH_THREADS="$MIID_CPU_THREADS" MIID_BG_CHECK_THREADS="$MIID_CPU_THREADS"
             OMP_NUM_THREADS="$MIID_CPU_THREADS" OPENBLAS_NUM_THREADS="$MIID_CPU_THREADS"
             MKL_NUM_THREADS="$MIID_CPU_THREADS")
fi
env HF_TOKEN="$HF_TOKEN" \
    FLUX_DEVICE="${FLUX_DEVICE:-cuda}" \
    MIID_ARCHIVE_IMAGES="${MIID_ARCHIVE_IMAGES:-1}" \
    MIID_OFFLOAD_MODE="${MIID_OFFLOAD_MODE:-mapped}" \
    MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${threads[@]}" \
    pm2 start "${cmd[0]}" --name "$NAME" --cwd "$REPO" --interpreter none --time -- "${cmd[@]:1}"

pm2 save >/dev/null 2>&1 || true
echo
echo "Started. Watch it with:  pm2 logs $NAME --lines 50"
