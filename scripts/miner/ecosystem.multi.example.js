// pm2 ecosystem for several hotkeys on one machine — see docs/miner.md,
// "Running several hotkeys", and read the duplication policy there first.
//
//   cp scripts/miner/ecosystem.multi.example.js ecosystem.config.js
//   # edit HOTKEYS, COLDKEY, HF_TOKEN; then
//   pm2 start ecosystem.config.js
//
// One process per hotkey, each on its own axon port. All of them share the
// sibling ledger (no two submit near-identical takes) and the GPU lock (they
// take turns on the card and see each other's queues). One 16 GB GPU serves
// about two hotkeys with retries; add GPUs, not hotkeys, beyond that.

const path = require("path");

const REPO = __dirname.replace(/\/scripts\/miner$/, "");
const COLDKEY = "my_coldkey";
const HOTKEYS = ["hotkey_0", "hotkey_1"];
const FIRST_AXON_PORT = 8092;
// CPU threads per process: the cgroup quota divided by the number of miners.
const THREADS = 4;
const SHARED = path.join(REPO, "..", "miid_shared");

const common = {
  cwd: REPO,
  interpreter: "none",
  autorestart: true,
  max_restarts: 50,
  kill_timeout: 1200000, // let an in-flight round finish before SIGKILL
  env: {
    HF_TOKEN: "hf_YOUR_TOKEN_HERE",
    FLUX_DEVICE: "cuda",
    MIID_ARCHIVE_IMAGES: "1",
    MIID_SIBLING_DIR: path.join(SHARED, "siblings"),
    MIID_GPU_LOCK_DIR: path.join(SHARED, "gpu"),
    // Two processes polishing against each other is wasted time on a shared card.
    MIID_POLISH_ATTEMPTS: "1",
    // Memory: keep the FLUX weights memory-mapped (one shared copy in the
    // page cache, ~1.5 GB private per process instead of 15 GB), limit glibc
    // to two malloc arenas, and size every CPU thread pool to this process's
    // share of the cgroup quota rather than the host's core count.
    MIID_OFFLOAD_MODE: "mapped",
    MALLOC_ARENA_MAX: "2",
    MIID_TORCH_THREADS: String(THREADS),
    MIID_BG_CHECK_THREADS: String(THREADS),
    OMP_NUM_THREADS: String(THREADS),
    OPENBLAS_NUM_THREADS: String(THREADS),
    MKL_NUM_THREADS: String(THREADS),
    TOKENIZERS_PARALLELISM: "false",
  },
};

module.exports = {
  apps: HOTKEYS.map((hotkey, i) => ({
    ...common,
    name: `miid_miner_${i}`,
    script: path.join(REPO, "miner_env", "bin", "python"),
    args: [
      "neurons/miner.py",
      "--netuid", "54",
      "--subtensor.network", "finney",
      "--subtensor.chain_endpoint", "wss://entrypoint-finney.opentensor.ai:443",
      "--wallet.name", COLDKEY,
      "--wallet.hotkey", hotkey,
      "--axon.port", String(FIRST_AXON_PORT + i),
      "--logging.debug",
    ].join(" "),
  })),
};
