#!/usr/bin/env bash
# Start one pm2 miner process per hotkey on this machine, all sharing the GPU
# and the sibling ledger (docs/miner.md, "Running several hotkeys" — read the
# duplication policy there first).
#
#   bash scripts/miner/start_hotkeys.sh                       # all six: hotkey_0:8092 … hotkey_5:8097
#   bash scripts/miner/start_hotkeys.sh hotkey_4:8096 hotkey_5:8097   # add or replace just these
#   bash scripts/miner/start_hotkeys.sh --dry-run             # print the pm2 commands only
#
# Each argument is <hotkey>:<axon port>. The public port is taken from
# VAST_TCP_PORT_<axon port> (Vast.ai) when that variable exists, else the
# axon port itself. A hotkey that already has a pm2 process (name
# miid_miner_<n>, or miid_miner for hotkey_0) is deleted and started again
# with the shared settings — that is a restart of a live miner, so run it
# right after a "Request completed" line in its log. CPU threads per process
# are 4, or the cgroup quota divided by the number of miner processes on the
# box after the run when that is larger (MIID_CPU_THREADS overrides).
#
# Capacity, measured 2026-10-16 on one 16 GB card before that day's changes
# (~17 s per attempt): four hotkeys kept the GPU ~78% busy (190 attempts and
# 26 requests an hour, 1.1 retries per round); six saturated it (first
# attempts waited up to 1062 s of the 1200 s, 4 of 70 rounds short of a
# slot). The changes cut the GPU seconds per request — one text-encoder
# pass per request, transformer and VAE resident across its attempts, the
# judging overlapped with the next generation; see docs/miner.md,
# "Capacity" — and retries are off by default in favour of polish. Watch
# "waited ...s for the GPU" and "Generated N variations" in the logs; add
# GPUs, not hotkeys, when first attempts wait more than ~600 s.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${MINER_PYTHON:-$REPO/miner_env/bin/python}"
COLDKEY="${MIID_COLDKEY:-phantom_coldkey}"
NETUID="${MIID_NETUID:-54}"
NETWORK="${MIID_NETWORK:-finney}"
ENDPOINT="${MIID_CHAIN_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"
SHARED="${MIID_SHARED_DIR:-/workspace/miid_shared}"
EXTERNAL_IP="${MIID_EXTERNAL_IP:-${PUBLIC_IPADDR:-}}"
THREADS="${MIID_CPU_THREADS:-}"

DRY_RUN=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n 2,22p "${BASH_SOURCE[0]}"; exit 0 ;;
        *) ARGS+=("$a") ;;
    esac
done
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(hotkey_0:8092 hotkey_1:8093 hotkey_2:8094 hotkey_3:8095 hotkey_4:8096 hotkey_5:8097)
fi

# nvm's node/pm2 are only on PATH in login shells.
if ! command -v pm2 >/dev/null 2>&1 && [[ -f /opt/nvm/nvm.sh ]]; then
    # shellcheck disable=SC1091
    . /opt/nvm/nvm.sh
fi
command -v pm2 >/dev/null 2>&1 || { echo "pm2 not found on PATH" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "miner python not found at $PYTHON (set MINER_PYTHON)" >&2; exit 1; }

# The Hugging Face token: environment, else ${WORKSPACE}/.env.
if [[ -z "${HF_TOKEN:-}" && -f "${WORKSPACE:-/workspace}/.env" ]]; then
    HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "${WORKSPACE:-/workspace}/.env" | tr -d '"' | head -1)"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set (export it, or put HF_TOKEN=... in ${WORKSPACE:-/workspace}/.env)" >&2; exit 1; }
[[ -n "$EXTERNAL_IP" ]] || { echo "External IP unknown: set MIID_EXTERNAL_IP (PUBLIC_IPADDR is empty)" >&2; exit 1; }

mkdir -p "$SHARED/siblings" "$SHARED/gpu"

# CPU threads per miner process: 4, or the cgroup quota (cores) divided by
# the number of miner processes on the box after this run — those already
# under pm2 plus the ones started here — when that is larger. The pools only
# work in short CPU phases (checks ~2 s an attempt, voice ~5 s a request,
# the face blend in its own thread), so six miners at 4 threads on a 15-core
# quota measured 1.6 cores in use on average and no quota throttling at all;
# the GPU is the limit, not the CPU. Fewer than 4 only makes those phases
# slower for nothing.
if [[ -z "$THREADS" ]]; then
    cores=$(nproc)
    if read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null && [[ "$quota" != "max" && -n "${period:-}" ]]; then
        cores=$(( quota / period ))
    fi
    names="$(pm2 jlist 2>/dev/null | grep -o '"name":"miid_miner[^"]*"' || true)"
    for spec in "${ARGS[@]}"; do
        hotkey="${spec%%:*}"
        suffix="${hotkey##*_}"
        if [[ "$hotkey" == "hotkey_0" ]]; then names+=$'\n''"name":"miid_miner"'; else names+=$'\n'"\"name\":\"miid_miner_${suffix}\""; fi
    done
    n=$(printf '%s\n' "$names" | sed '/^$/d' | sort -u | wc -l)
    THREADS=$(( (cores + n / 2) / n ))
    (( THREADS >= 4 )) || THREADS=4
    echo "CPU threads per process: $THREADS ($cores cores for $n miner processes)"
fi

for spec in "${ARGS[@]}"; do
    hotkey="${spec%%:*}"
    port="${spec##*:}"
    [[ "$hotkey" != "$spec" && "$port" =~ ^[0-9]+$ ]] || { echo "bad argument '$spec' (want <hotkey>:<port>)" >&2; exit 1; }
    [[ -f "$HOME/.bittensor/wallets/$COLDKEY/hotkeys/$hotkey" ]] || { echo "no key file for $COLDKEY/$hotkey" >&2; exit 1; }
    public_var="VAST_TCP_PORT_${port}"
    public_port="${!public_var:-$port}"
    suffix="${hotkey##*_}"
    if [[ "$hotkey" == "hotkey_0" ]]; then name="miid_miner"; else name="miid_miner_${suffix}"; fi

    cmd=("$PYTHON" neurons/miner.py
        --netuid "$NETUID" --subtensor.network "$NETWORK" --subtensor.chain_endpoint "$ENDPOINT"
        --wallet.name "$COLDKEY" --wallet.hotkey "$hotkey"
        --axon.port "$port" --axon.ip 0.0.0.0
        --axon.external_ip "$EXTERNAL_IP" --axon.external_port "$public_port"
        --logging.debug)

    echo "== $name: $hotkey on :$port (public $EXTERNAL_IP:$public_port)"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "   pm2 start ... -- ${cmd[*]}"
        continue
    fi
    if pm2 describe "$name" >/dev/null 2>&1; then
        echo "   replacing existing pm2 process $name"
        pm2 delete "$name" >/dev/null
    fi
    # Memory: mapped offload keeps the FLUX weights in the shared page cache
    # (~1.5 GB private per process instead of 15 GB, no copy-back churn);
    # MALLOC_ARENA_MAX stops glibc giving each of the hundreds of threads its
    # own 64 MB arena; the thread caps size every CPU pool to this process's
    # share of the cgroup quota instead of the host's 128 cores.
    HF_TOKEN="$HF_TOKEN" \
    FLUX_DEVICE="${FLUX_DEVICE:-cuda}" \
    MIID_ARCHIVE_IMAGES="${MIID_ARCHIVE_IMAGES:-1}" \
    MIID_SIBLING_DIR="$SHARED/siblings" \
    MIID_GPU_LOCK_DIR="$SHARED/gpu" \
    MIID_POLISH_ATTEMPTS="${MIID_POLISH_ATTEMPTS:-2}" \
    MIID_OFFLOAD_MODE="${MIID_OFFLOAD_MODE:-mapped}" \
    MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}" \
    MIID_TORCH_THREADS="$THREADS" \
    MIID_BG_CHECK_THREADS="$THREADS" \
    OMP_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    pm2 start "${cmd[0]}" --name "$name" --cwd "$REPO" --interpreter none --time -- "${cmd[@]:1}"
done

if [[ $DRY_RUN -eq 0 ]]; then
    pm2 save >/dev/null 2>&1 || true
    echo
    echo "Started. Watch one with:  pm2 logs ${name} --lines 50"
    echo "Each process logs its diversity profile, 'sibling ledger: $SHARED/siblings' and 'GPU sharing: file lock in $SHARED/gpu' at start."
fi
