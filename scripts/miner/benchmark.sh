#!/usr/bin/env bash
# Predict validation scores for archived miner submissions.
#
#   bash scripts/miner/benchmark.sh                 # every archived round with images
#   bash scripts/miner/benchmark.sh --last 10 --html bench.html
#   bash scripts/miner/benchmark.sh --help
#
# The benchmark needs MediaPipe, which miner_env does not ship. Installing it
# into miner_env would change the environment a live miner is running from,
# so this keeps a separate bench_env/ that sees miner_env's packages (torch,
# transformers, AdaFace deps) through a .pth hook and adds only MediaPipe.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINER_ENV="${MINER_ENV:-$REPO/miner_env}"
BENCH_ENV="${BENCH_ENV:-$REPO/bench_env}"

if [[ ! -x "$MINER_ENV/bin/python" ]]; then
    echo "miner_env not found at $MINER_ENV — run scripts/miner/setup.sh first (or set MINER_ENV)." >&2
    exit 1
fi

if [[ ! -x "$BENCH_ENV/bin/python" ]] || ! "$BENCH_ENV/bin/python" -c "import mediapipe" 2>/dev/null; then
    echo "Setting up $BENCH_ENV (one time) ..."
    "$MINER_ENV/bin/python" -m venv "$BENCH_ENV"
    miner_site="$("$MINER_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    bench_site="$("$BENCH_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    # addsitedir (not a bare path) so miner_env's own .pth files — including
    # the editable install of this repo — are processed too.
    echo "import site; site.addsitedir('$miner_site')" > "$bench_site/zz_miner_env.pth"
    "$BENCH_ENV/bin/pip" install --quiet --disable-pip-version-check mediapipe
fi

cd "$REPO"
exec "$BENCH_ENV/bin/python" -m MIID.benchmark "$@"
