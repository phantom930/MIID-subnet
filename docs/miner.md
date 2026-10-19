# Yanez Miner

This document explains how to run a Yanez miner on the Bittensor network. The Yanez subnet (Subnet 54) focuses on **identity-preserving face image variations** for KYC and fraud-detection research, with a **voice clone** challenge collected for UAV post-grading.

## Overview

Yanez miners receive **face image variation requests** from validators:

- Receive a base face image and variation requirements
- **Variation types (current):** `background_in`, `background_out`, three combined edits (`lighting_edit+expression_edit`, `lighting_edit+pose_edit`, `pose_edit+expression_edit`), and synthetic **`screen_replay`** (device named + ≥2 visual cues; iterate against Passive Liveness v3 `/is_live`)
- Generate identity-preserving image variations using a diffusion model
- Encrypt and upload results to S3
- Return signed submission references to the validator

Real IOTD / physical screen-replay is paused for this cycle (protocol fields commented out).
Validators may also attach a **voice request** in the same round (reference WAV + target speech text). Implement `generate_voice_clone` in `MIID/miner/voice_generator.py` to submit a cloned WAV. Voice is UAV post-graded only (not part of live KAV image rewards).

**Image generation is the KAV-scored task.** You need a GPU and the image-generation stack to earn live image rewards. Voice does not replace that path.

## Subnet Phases & Timeline

| Date | Milestone |
|------|-----------|
| **Sep 24 – Oct 15, 2026** | **Phase 5 Cycle 1 Sandbox (current; first official Phase 5 cycle). Calibration + stability testing. Voice clone challenge (UAV post-graded only) + continued face image KAV path. Burn ratio 35%.** |
| Oct 15 – Nov 16, 2026 | Phase 5 Cycle 1 Execution goes live (~1 month). Reward allocation uses validated reputation alongside the ongoing online flow. Burn ratio stays 35%. |
| Nov 16 – Nov 30, 2026 | Phase 5 Cycle 2 Sandbox (calibration + stability testing). Normal KAV online flow unchanged. Burn ratio 35%. |

For the full roadmap and detailed architecture, see [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf).

## Requirements

- Python 3.10 or higher
- A Bittensor wallet with TAO for registration
- A Hugging Face account with an API token (free)
- Access to the base image models used by the miner:
  - [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (used by `flux_klein`)
  - [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (used by `pulid` fallback and `flux_kontext` alternative)
  - Optional alternative: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- **GPU with at least 8GB VRAM** (NVIDIA CUDA or Apple Silicon MPS) — required for production mining
- CPU: 8 physical cores recommended (12+ preferred when handling image jobs continuously)
- At least 16GB RAM (32GB+ recommended)
- Storage: 80GB minimum free disk, 150GB+ recommended (OS + repo + venv + diffusion model weights + cache + outputs)
- Base Python packages from `requirements.txt` plus miner image packages from `requirements-miner.txt` (`torch`, `diffusers`, `transformers`, `Pillow`, `opencv-python`, etc.)
- Open port 8091 for validator communication ([Network Setup Guide](network_setup.md))

> **Machine sizing:** Target at least **8 cores / 16GB RAM / 80GB free disk + 8GB VRAM GPU** (better: 12+ cores / 32GB RAM / 150GB+ disk + higher-VRAM GPU).

---

## Installation

**If you are not used to the command line or Python**, use **Option 1**. It runs one script that installs dependencies for you.

**If you are comfortable with Linux/macOS, Python virtual environments, and package managers**, use **Option 2** (step-by-step guide below).

### Option 1: Automated setup (recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

2. **Run the setup script** (targets **Ubuntu/Debian** with `apt`; installs system packages, Python venv, image-generation dependencies, AdaFace, and more):
```bash
bash scripts/miner/setup.sh
```

On **macOS** (or if you prefer not to use `apt`), use **Option 2** instead.

3. **Activate the virtual environment** (do this every time you open a new terminal before running the miner):
```bash
source miner_env/bin/activate
```

4. **Set your Hugging Face token and GPU device** before the first miner run (create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept model licenses):
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"   # or mps (Apple) or cpu (slow, not recommended)
```
If the script could not pre-download FLUX weights, run `python -m MIID.miner.downloading_model` after `HF_TOKEN` is set.

5. **Create a wallet and register** (needs a small amount of TAO; skip if you already have a registered miner). For first-time setup, we recommend starting on testnet first (see [First-time setup: run on testnet first](#first-time-setup-run-on-testnet-first)):
```bash
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
```

6. **Start the miner**
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug
```

You should see `Phase 4 image generation: ENABLED` in the logs. If you see `DISABLED`, complete the [troubleshooting steps](#phase-4-image-generation-disabled-missing-packages) below — without image generation enabled you will not earn rewards.

### First-time setup: run on testnet first

For first-time setup, testnet is the safest place to validate your miner end-to-end before running on mainnet.  
Our Yanez testnet netuid is **`322`**.

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 322 --subtensor.network test --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug --axon.port <YOUR_OPEN_PORT> --axon.ip 0.0.0.0 --axon.external_ip <YOUR_PUBLIC_IP> --axon.external_port <YOUR_PUBLIC_PORT>
```

View your testnet results on [Weights & Biases (Yanez subnet 322)](https://wandb.ai/MIID-dev-test/subnet322-test/table?nw=nwuseraromanhigh).

Optional walkthrough video (different subnet, but setup flow is very similar): [YouTube guide](https://www.youtube.com/watch?v=UH_sOZSIk10&list=PLqsRtfujbWUklHVRqOAHQ7tzXYZ8OAZlR).

Flag breakdown for the command above:

- `--netuid 322`: selects the subnet to join; use `322` for Yanez testnet.
- `--subtensor.network test`: points Bittensor to the test network instead of finney/mainnet.
- `--subtensor.chain_endpoint wss://test.finney.opentensor.ai:443`: sets the websocket RPC endpoint for chain communication.
- `--wallet.name miner_wallet`: chooses the coldkey wallet name that holds your miner identity.
- `--wallet.hotkey miner_hotkey`: chooses the hotkey used by the miner process to sign and serve requests.
- `--logging.debug`: enables verbose logs to make first-run troubleshooting easier.
- `--axon.port <YOUR_OPEN_PORT>`: local listening port where your miner axon serves validator requests.
- `--axon.ip 0.0.0.0`: binds the service to all local interfaces so it can accept incoming traffic.
- `--axon.external_ip <YOUR_PUBLIC_IP>`: advertises the public IP validators should use to reach your miner.
- `--axon.external_port <YOUR_PUBLIC_PORT>`: advertises the public port validators should use (often your forwarded/NAT port).

### Option 2: Manual installation

Follow the **[step-by-step guide](#step-by-step-guide-manual-installation)** from **Step 1** through **Step 12**, then start the miner.

### Installation recommendations

- **Non-technical users**: Prefer **Option 1**, then Steps 3–6 above (activate → env vars → wallet/register → run).
- **Technical users**: Use **Option 2** if you want full control, custom paths, or a non-Ubuntu setup.
- **Production**: After either path, consider a process manager (systemd, supervisor, tmux, pm2 — see [Background Mining Guide](background_mining.md)), monitoring, and logging.

---

## Step-by-step guide (manual installation)

Use this section when you chose **Option 2**, or when you need to **verify or redo** a single step.

### Step 1: Install System Dependencies

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake git curl wget jq nano
```

**macOS:**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python3 git curl wget jq cmake
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
```

From this point on, **all commands are run from inside the `MIID-subnet` folder**.

### Step 3: Create and Activate a Virtual Environment

```bash
python3 -m venv miner_env
source miner_env/bin/activate
```

**Every time you open a new terminal**, activate again:

```bash
cd MIID-subnet
source miner_env/bin/activate
```

### Step 4: Install Base Python Dependencies

```bash
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e .
```

### Step 5: Install Miner Image-Generation Packages

```bash
pip install -r requirements-miner.txt
```

Validators do **not** need `requirements-miner.txt` — only miners.

**Optional:** `pip install timelock` (drand encryption; otherwise sandbox raw-bytes fallback).

### Step 6: Create a Hugging Face Account and Token

1. [huggingface.co/join](https://huggingface.co/join)
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a **Read** token (`hf_...`)
3. Accept licenses: [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), and optionally [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
4. Set the token:

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

> To persist: add to `~/.bashrc` or `~/.zshrc`.

### Step 7: Set Up AdaFace (Face Identity Validation)

```bash
git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace
mkdir -p MIID/miner/AdaFace/pretrained
gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
```

Manual download: [Google Drive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) → place at `MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt`.

### Step 8: Set Your GPU Device

```bash
export FLUX_DEVICE="cuda"    # NVIDIA
# export FLUX_DEVICE="mps"   # Apple Silicon
# export FLUX_DEVICE="cpu"    # slow — not viable for production mining
```

### Step 9: Pre-Download the Diffusion Model (Recommended)

```bash
python -m MIID.miner.downloading_model
```

> **Note:** FLUX.2-klein is the default baseline in code. You can experiment with other diffusion setups in `MIID/miner/generate_variations.py` (model list and `MIID_MODEL`). Better models can give you an edge.

### Step 10: Create a Bittensor Wallet

```bash
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
```

**Write down and securely store your mnemonic seed phrases.**

### Step 11: Register Your Miner on the Subnet

```bash
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
```

### Step 12: Start the Miner

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py \
  --netuid 54 \
  --subtensor.network finney \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --wallet.name miner_wallet \
  --wallet.hotkey miner_hotkey \
  --logging.debug
```

The miner logs `Phase 4 image generation: ENABLED`, then loads `flux_klein` (or the model `MIID_MODEL` selects).

### Quick Start Cheat Sheet

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake git curl wget jq nano
git clone https://github.com/yanez-compliance/MIID-subnet.git
cd MIID-subnet
python3 -m venv miner_env
source miner_env/bin/activate
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e .
pip install -r requirements-miner.txt
pip install timelock      # optional
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
git clone https://github.com/mk-minchul/AdaFace.git MIID/miner/AdaFace
mkdir -p MIID/miner/AdaFace/pretrained
gdown 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI -O MIID/miner/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
export FLUX_DEVICE="cuda"
python -m MIID.miner.downloading_model
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey
btcli subnet register --netuid 54 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --subtensor.network finney
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name miner_wallet --wallet.hotkey miner_hotkey --logging.debug
```

---

## Running a Miner (register + start)

If you **already created a wallet and registered**, skip step 1 below.

1. Register your miner to the subnet (skip if already registered):
```bash
btcli subnet register --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney
```

2. Start your miner:
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export FLUX_DEVICE="cuda"
python neurons/miner.py --netuid 54 --subtensor.network finney --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --logging.debug
```

For detailed information about logging setup and management, see the [Logging Guide](logging.md).

For running your miner in the background (recommended for production), see the [Background Mining Guide](background_mining.md).

## Running several hotkeys

**Read the policy first.** The [Face Variation Reward System](Face%20Variation%20Reward%20System.pdf) ("Duplicate Submissions and Multi-Hotkey Activity") describes a cross-miner duplication analysis run during post-validation. In Cycle 1 one operator ran five hotkeys from the same model; they placed 2nd to 6th on the leaderboard, were found to be "submitting nearly identical images", and all five were moved to the Watch tier with reputation near zero. The document states that "miners are strongly encouraged to operate a single hotkey", that "submitting the same model outputs across multiple hotkeys will be treated as duplication", and that the one-hotkey-keeps-its-score leniency "should not be assumed in future cycles". Hotkeys under one coldkey were grouped together in that analysis. Nothing in this repository changes that policy; what follows only makes sure that hotkeys you do run never submit near-identical images.

**Capacity.** Every validator queries every hotkey (the default `--neuron.sample_size` is 250 against 256 UIDs), so each hotkey receives the same 6–7 requests an hour, and the requests of one validator round reach all of them within seconds. Measured on 2026-10-16 on one 16 GB card with the code as it was that morning (~17 s per attempt, every model uploaded again for every attempt, the take judged while the GPU waited): four hotkeys kept the card 78% busy — 190 attempts and 26 requests an hour, 1.1 retries per round, no slot ever missing; six hotkeys saturated it — requests waited 584 s on average and up to 1062 s of the 1200 s for their first attempt, retries fell to 0.3 per round, polish to nothing, 4 of 70 rounds went out short of a slot and two voice clones were skipped. What a GPU can carry is therefore decided by the GPU seconds one request's first pass costs, which is what the changes of that day cut: one text-encoder pass per request instead of one per attempt, the transformer and VAE resident across the attempts of a request, judging overlapped with the next generation (see "Request timing"). Beyond what one card carries, add GPUs, not hotkeys.

**Memory.** With the defaults each process keeps about 1.5 GB of private memory for the FLUX.2 Klein pipeline: diffusers memory-maps the safetensors files, and `MIID_OFFLOAD_MODE=mapped` uploads each model to the GPU from those mapped tensors when it runs, keeps it there (within `MIID_RESIDENT_GB`) until the process hands the GPU to another request or process, and evicts it without copying back, so the 15 GB of weights live once in the page cache and every process on the machine shares them. (With diffusers' own model offload, every generation copied all three models back into fresh private memory: four miners on one box were measured at 40–55 GB RSS each, growing by several GB an hour.) The voice models are kept on the CPU tensors they were loaded with (~3 GB private) rather than re-allocated each round, `MALLOC_ARENA_MAX=2` stops glibc giving each of the hundreds of threads its own arena, the heap is trimmed after every request, and every CPU thread pool is sized to `MIID_TORCH_THREADS` — set that to the cgroup quota divided by the number of miners (`scripts/miner/start_hotkeys.sh` uses 4, or the cgroup quota divided by the number of miner processes on the box when that is larger; the pools only work in short CPU phases, and six miners at 4 threads on a 15-core quota measured 1.6 cores in use on average with no quota throttling, the GPU being the limit). Budget roughly 7 GB private memory per process plus one shared 18 GB of mapped weights, and watch the `process RSS` figure logged with every `Request completed` line.

Three pieces make several hotkeys safe to run from this code:

1. **A style profile per hotkey** (`MIID/miner/diversity.py`, on by default). Derived from the hotkey address: its own four indoor and four outdoor scenes, a preferred side for light and head turns, a warm or cool colour cast, a far-expression flavour and accessory colours, appended to the prompt as a short style clause. Two hotkeys therefore draw from different distributions, not just different seeds. `MIID_DIVERSITY=0` turns it off.
2. **A shared sibling ledger** (`MIID/miner/sibling_registry.py`, `MIID_SIBLING_DIR`). Every attempt's 256-bit dHash and CLIP embedding are written to `<dir>/<challenge_id>/`; before a take is accepted it is compared with the other hotkeys' takes of the same slot, and one within `MIID_DUP_DHASH_BITS` (16) or at CLIP cosine `MIID_DUP_CLIP_COSINE` (0.99) or above is ranked as a miss and re-rolled with a hint to compose differently. Use a local directory when the hotkeys share a machine and a network mount when they do not. Archived variations sit 68–120 bits from their own seed image, so honest renders with fresh seeds never trip this; it exists to guarantee that.
3. **GPU sharing between processes** (`MIID_GPU_LOCK_DIR`). Each hotkey runs its own `neurons/miner.py`; with this set they take turns on the GPU through a file lock in the same order as within one process — every first pass, then identity rescues, then edit retries and polish, earliest deadline first within each level — and see each other's queued requests, so one process's retries and polish yield to another process's waiting validator. (Without the phase rule, on 2026-10-14 one hotkey's eleven retries ran for 300 s while another hotkey's request waited 870 s and then had no time for a single attempt.) The voice models are parked in CPU memory after each request already; the idle GPU footprint per process is the CUDA context, the VAE and the face models, about 1 GB.

Example with pm2, one process per hotkey, distinct axon ports:

```bash
export MIID_SIBLING_DIR=/workspace/miid_shared/siblings
export MIID_GPU_LOCK_DIR=/workspace/miid_shared/gpu
for i in 0 1; do
  pm2 start bash --name "miid_miner_$i" -- -c "python neurons/miner.py --netuid 54 --subtensor.network finney \
    --wallet.name my_coldkey --wallet.hotkey hotkey_$i --axon.port $((8092 + i)) --logging.debug"
done
```

`scripts/miner/ecosystem.multi.example.js` is the same as a pm2 ecosystem file. Each process logs its profile at start (`diversity profile for 5Hpop...: side left, warm cast, ...`), the ledger path and the GPU sharing mode; a sibling collision shows as `near-identical to sibling 5Abc... attempt 1 (dHash 9 bits, CLIP cosine 0.994) — retrying at full strength with hints (duplicate: vary).`

---

## Dry Run: Check Your Miner Can Produce a Submission

Before a validator ever queries you, run the submission pipeline offline against a local face image.
It uses the same code path a real query does — generate variations, AdaFace identity check, drand
timelock encryption, upload — and tells you exactly which stage fails if one does:

```bash
# Simplest: local storage, throwaway key, the standard 5-variation challenge set
python -m MIID.miner.dry_run_submission --image /path/to/face.png

# Use your real hotkey so signatures and the S3 key path match production,
# and keep the generated images so you can look at them
python -m MIID.miner.dry_run_submission --image /path/to/face.png \
    --wallet-name your_wallet --wallet-hotkey your_hotkey \
    --save-images ./dryrun_images --output ./dryrun_submissions.json
```

Output is a per-variation table:

```
  [OK  ] background_in   612847 bytes  encrypted
           submissions/dryrun_1761402000/5Fxx.../a1b2c3d4e5f6a7b8/face/background_in_1761402000.png.tlock
  [DROP] pose_edit — AdaFace similarity below 0.4
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--variations "pose_edit:far,lighting_edit+expression_edit:medium"` | Test specific variations instead of the standard set |
| `--save-images DIR` | Write the unencrypted variations out so you can inspect quality |
| `--skip-identity-check` | Show what the model produced before AdaFace filtering (diagnostic only) |
| `--no-encrypt` | Skip drand timelock — sandbox only, real submissions must be encrypted |
| `--s3` | Upload to the real bucket instead of local storage (requires a real wallet) |
| `--debug` | Full bittensor debug logging |

By default nothing touches the shared bucket: encrypted output goes to `MIID_LOCAL_STORAGE`
(`/tmp/miid_submissions`). The command exits non-zero if it produced no submissions, so it also works
as a post-setup smoke check.

---

## Request Archive: What Did I Actually Submit?

Every validator approach is written to disk — what was asked for, what you sent back, and why
anything was dropped. Records are written whatever the outcome, including requests you refused
or rounds that crashed, so a quiet miner can be diagnosed after the fact instead of by tailing
logs and hoping.

Records land **inside your clone**, in `miner_requests/` (gitignored), one directory per request:

```
MIID-subnet/miner_requests/2026-09-29/20260929T014500Z__challenge_1759000000_5Dvgtk1b__49b25272/record.json
```

```jsonc
{
  "outcome": "submitted",           // or empty / error / no_image_request / phase4_unavailable
  "validator": { "hotkey": "5Dvg...", "name": "RoundTable21" },
  "request": {
    "challenge_id": "challenge_1759000000_5Dvgtk1b",
    "variation_requests": [ /* every type, intensity and prompt detail you were sent */ ],
    "base_image": { "sha256": "3a9aa2...", "bytes": 812345 },
    "daily_seed": { "filename": "844c449c32df_f_doc.png", "date": "2026-09-28" }
  },
  "generation": { "model_key": "pulid", "min_similarity": 0.4, "requested": 5, "submitted": 4 },
  "variation_report": [
    { "variation_type": "background_in", "status": "submitted", "model": "pulid", "s3_key": "..." },
    { "variation_type": "pose_edit+expression_edit", "status": "dropped", "model": "pulid",
      "reason": "AdaFace similarity below 0.4" }
  ],
  "submissions": [ /* the exact S3Submission list the validator received */ ],
  "screen_replay": { "attached": true, "capture_variant": "seed_smiling" }
}
```

`generation.model_key` records which approach ran, so you can compare how `flux_klein` and the
models `MIID_MODEL` selects are actually performing for you across rounds rather than guessing.

| Variable | Description | Default |
|----------|-------------|---------|
| `MIID_REQUEST_ARCHIVE` | Directory for records, used as given | `<repo>/miner_requests` |
| `MIID_ARCHIVE_ENABLED` | Set to `0` to turn archiving off entirely | on |
| `MIID_ARCHIVE_IMAGES` | Set to `1` to also keep the base image, both seeds, the generated variations, and the voice reference and clone WAVs (`media/voice_reference__<file>` and `media/voice_clone.wav`) | off |
| `MIID_ARCHIVE_MAX_RECORDS` | Records kept before the oldest are pruned (`0` = unlimited) | 500 |
| `MIID_ARCHIVE_RETENTION_HOURS` | Records older than this many hours are deleted when a new one is written (`0` = no age limit) | 24 |

**Media is not kept by default.** A round's variations run to tens of megabytes and you answer a
validator roughly hourly, so `MIID_ARCHIVE_IMAGES=1` will grow fast — turn it on while tuning
quality, then off again. Metadata-only records are a few KB each.

The dry run archives its runs the same way, so offline experiments sit in the same history:

```bash
python -m MIID.miner.dry_run_submission --image face.png              # archived like a real round
python -m MIID.miner.dry_run_submission --image face.png --no-archive # or don't
```

Archiving never affects what a validator receives: if a record cannot be written, the failure is
logged and the response goes out regardless.

---

## Benchmark: What Score Will My Submissions Get?

The grading API never tells a miner its per-variation `validation_score`. The benchmark
predicts it offline from archived rounds: it re-measures each submitted image the way the
published score sheet describes (`docs/Face Variation Reward System.pdf`, Cycle 4) and maps the
result to a score from −5 to 5. It also reports what that score means for reward. Records need
images, so run the miner with `MIID_ARCHIVE_IMAGES=1`.

```bash
bash scripts/miner/benchmark.sh                          # every archived round with images
bash scripts/miner/benchmark.sh --last 10 --html bench.html -v
bash scripts/miner/benchmark.sh --base face.png --image out.png \
    --type lighting_edit+pose_edit --intensity medium+far  # one image outside the archive
```

The first run creates `bench_env/`, which uses `miner_env`'s packages and adds MediaPipe. It
then downloads the MediaPipe task files and CLIP into `~/.cache/miid_benchmark` and the
Hugging Face cache. Nothing is installed into `miner_env`, and everything runs on CPU by default
(`--device cuda` to share the miner's GPU). Measurements are cached per record in
`benchmark.json`, so a re-run only re-scores.

Per variation it prints the predicted score, AdaFace identity, Δrep, and the main reason. For
each round it prints:

- **Expected KAV.** The validator grades one random slot per round and gives it 0 unless that
  slot's identity is at least 0.6, so this is the mean over slots of `score / 5` for slots that
  pass that gate.
- **Δrep if reviewed.** The reputation change from the manual-review table (+0.10 for a 5,
  −0.10 for a −3). Reputation (UAV) carries 90% of miner emissions.

The summary breaks scores down by model and by slot, so you can compare `flux_klein` against
the models `MIID_MODEL` selects on your own submissions.

| Check | Score | Basis |
|-------|-------|-------|
| AdaFace identity < 0.4, face count ≠ 1, gender-mismatched religious covering | −3 | score sheet |
| Face > 15.5% narrower than the seed | −4 | score sheet |
| Face pixels copied from the seed | −5 | sheet; detection threshold estimated |
| Duplicate of the seed / resolution ≤ 512×672 / face not dominant | −1 | sheet; dominance threshold estimated |
| Claimed variation not detected | −2 | sheet; detector thresholds estimated |
| Every requested component at the requested intensity | 5 | sheet; intensity bins estimated |
| Partial match, wrong intensity, or an unrequested extra variation | 3 | sheet |
| Identity 0.4–0.6 | at most 1 | Cycle 1 flowchart |
| Background slot: accessory match + background changed + identity ≥ 0.6 | 5 (4 without background change) | score sheet |

These are predictions, not grades. The score sheet says *what* the grader checks but not the
numeric cut-offs its detectors use, so the pose, expression, lighting, background and accessory
thresholds were calibrated by eye against archived rounds. They are all in
`MIID/benchmark/scoring.py` (`T`), with the source of each rule.

---

## Configuration Options

Command-line arguments:

- `--neuron.logging.debug`: Enable debug logging
- `--neuron.log_responses`: Save miner responses for analysis
- `--neuron.response_cache_dir`: Directory to store response logs

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` or `HUGGINGFACE_TOKEN` | Your Hugging Face API token | none |
| `FLUX_DEVICE` | Device for image generation (`cuda`, `mps`, or `cpu`) | `cpu` |
| `MIID_MODEL` | Select `flux_klein`, `pulid`, `flux_kontext`, or `qwen`; an unknown name logs a warning and uses `flux_klein` | `flux_klein` |
| `MIID_INFERENCE_STEPS`, `MIID_GUIDANCE_SCALE` | Tune generation (see `MIID/miner/generate_variations.py`) | see code |
| `MIID_IDENTITY_TARGET` | AdaFace similarity a slot should reach before its take is accepted as-is: under it the slot is re-rolled while time allows (a rescue attempt with the edit held back when `MIID_IDENTITY_RETRIES` is set, otherwise a full-strength polish re-roll), and between two takes of one slot the one over the target wins before edit strength is compared. The validator's gate is 0.6, but the paid top 50 of a round are decided by identity around 0.8, so the default was raised from 0.6 on 2026-10-15 | 0.7 |
| `MIID_IDENTITY_RETRIES` | Extra attempts per slot that misses the identity target or a measured edit (identity rescues hold the edit back; edit retries carry a hint naming the miss). `0` since 2026-10-16: with six hotkeys on one 16 GB card the first passes fill it, and the time left goes to polish, which then also re-rolls the slots that missed a target (see `MIID_POLISH_ATTEMPTS`). Set 3 on a card with time to spare | 0 |
| `MIID_IDENTITY_GOAL` | Identity a slot that already meets its targets is polished toward: while no other request is queued, it is re-rolled at full edit strength and a higher-identity take is kept **only if every measured edit (head turn, lighting, expression, scene, accessory) is at least as strong as the incumbent's** — a higher-identity take is usually a weaker edit, and the validator ranks on the edit first, identity second | 0.85 |
| `MIID_POLISH_ATTEMPTS` | Full-strength re-rolls allowed per slot (never for `screen_replay`, only on GPU time no first pass or rescue needs): the satisfied slots short of `MIID_IDENTITY_GOAL`, lowest identity first, re-rolled on their first-pass prompt so that on FLUX.2 klein they upload no weights; and, when `MIID_IDENTITY_RETRIES` is 0, before them every slot that missed a target — a new seed at full strength (never the edit held back), a hint when an edit measured short, the better take kept by the usual ranking. `0` disables polishing | 2 |
| `MIID_FACE_BLEND` | Blend the base image's face back into a take whose identity is under `MIID_FACE_BLEND_GOAL`: the base face is warped onto the take's five MTCNN landmarks, colour-matched to the take, and blended in — over the visible face skin on background slots, with a local colour match that keeps the take's lighting on lighting slots, and over the brows, eyes and nose only on expression slots. The scene, hair, ears, headwear, lighting and mouth stay the take's own. The result is kept only if identity rose, no measured edit weakened (scene, accessory, lighting magnitude, expression), and the image is not a copy of the seed (dHash). Pose slots and any take whose face moved are left alone. During the first pass the blend runs in a background thread while the next slot generates. `0` disables it. See "Face blend" below | on |
| `MIID_FACE_BLEND_GOAL` | Identity the blend aims for; a background take already at it is left alone | 0.88 |
| `MIID_FACE_BLEND_MAX` | Strongest blend allowed (1.0 would paste the base face). 0.9 gained no identity over 0.8 on 80 archived takes but pulled the CLIP scene reading down twice as far; a blend that drops the wanted-scene probability by more than 0.25 is refused in any case | 0.8 |
| `MIID_FACE_BLEND_SLOTS` | Slot types the blend applies to (pose slots cannot be blended; the lighting+expression combo lost identity on every variant and is left out) | `background_in,background_out,lighting_edit,expression_edit` |
| `MIID_FACE_BLEND_MASK` | Region the base face is blended into: `skin` — the inner face (just above the brows to the chin, cheek to cheek) plus the connected skin around it that still resembles the base there (forehead below a brim, cheeks beside a hijab, jaw and chin), so fabric or a hat worn over the base's hair is never painted over; `ellipse` — the inner face only | `skin` |
| `MIID_FACE_BLEND_POISSON` | Seamless (Poisson) cloning before the mix. Hides the boundary but re-shades the face from the take's gradients, which AdaFace reads as a different face (0.84 against 0.88 identity on 40 archived takes), so it is off | off |
| `MIID_EDIT_CHECK` | Measure every attempt's edit in-round — lighting magnitude on the face (the benchmark's formula), expression and accessory with CLIP on the head crop — rank takes on those measurements before identity, and re-roll gross misses with a hint (no lighting change, a far lighting request that reads light, a medium/far expression that reads neutral, a missing or wrong accessory). `0` disables the lighting, expression and accessory measurements | on |
| `MIID_EXPR_NEUTRAL_PROB` | CLIP "neutral" probability above which a medium or far expression request counts as a miss | 0.6 |
| `MIID_BG_CHECK` | Check every background attempt's scene with CLIP (the benchmark's own environment prompts) and re-roll it with a hint when it reads as the wrong kind of scene. `0` disables the check. (The earlier "plain studio backdrop" rule is gone: it fired on passport framing, not on blank backdrops, and only burned retries) | on |
| `MIID_BG_CHECK_DEVICE` | Device for every CLIP check; CPU takes ~0.4 s per call and leaves the GPU to the image pipeline | `cpu` |
| `MIID_BG_CLIP` | CLIP checkpoint for the checks (cached on first use, ~1.7 GB) | `openai/clip-vit-large-patch14` |
| `MIID_BG_MISMATCH_PROB` | Probability at which the other kind of scene fails a background attempt | 0.5 |
| `MIID_BG_CHECK_THREADS` | CPU threads for the CLIP forward | min(16, CPU quota) |
| `MIID_DIVERSITY` | Give this hotkey its own style profile (scenes, side, colour cast, far-expression flavour, accessory colours) derived from its address, so several hotkeys do not draw from one distribution. See "Running several hotkeys" | on |
| `MIID_SIBLING_DIR` | Directory shared by the hotkeys of one operator: every attempt's dHash and CLIP embedding are recorded per challenge, and a take near-identical to another hotkey's is re-rolled. Unset disables the ledger | unset |
| `MIID_DUP_DHASH_BITS`, `MIID_DUP_CLIP_COSINE` | What counts as near-identical to a sibling's take: dHash within this many of 256 bits, or CLIP cosine at or above this | 16, 0.99 |
| `MIID_SIBLING_TTL_HOURS` | Challenges older than this are pruned from the ledger | 24 |
| `MIID_GPU_LOCK_DIR` | Directory through which several miner processes on one machine share the GPU (file lock) and see each other's queued requests | unset |
| `MIID_OFFLOAD_MODE` | How the image model's weights move between CPU memory and the GPU: `mapped` keeps them memory-mapped from the safetensors files (one shared copy in the page cache for every process on the machine, ~1.5 GB private memory per process instead of 15 GB; a model is uploaded when it runs, stays on the GPU within `MIID_RESIDENT_GB` until the process hands the GPU over, and is evicted without a copy back); `model` is diffusers' model offload (whole models, copied back into fresh private memory after every generation — the process grows by several GB an hour); `sequential` is diffusers' layer-by-layer offload (lowest VRAM, slowest). Cards under 12 GB use `sequential` unless set explicitly. Replaces `MIID_SEQUENTIAL_CPU_OFFLOAD` (`1` → sequential, `0` → model, still honoured) | `mapped` |
| `MIID_RESIDENT_GB` | Weights `mapped` offload keeps on the GPU between attempts, the largest evicted first when a model does not fit. 9 holds the klein transformer (7.8 GB) and VAE beside the denoising activations and five idle sibling processes on a 16 GB card; the 8 GB text encoder swaps with the transformer only when a prompt has to be encoded, which the first pass does once for all its slots (see "Request timing") | 9 |
| `MIID_RECYCLE_RESERVED_GB` | Idle GPU memory (torch reserved after emptying the cache) at which a miner process exits so pm2 restarts it with a clean card — only while no request is in flight or queued, 30 s after the last response, after 30 min uptime, and one hotkey per 10 min on the machine (`recycle.last` in `MIID_GPU_LOCK_DIR`). Guards against the per-request GPU memory creep that emptied every round on 2026-10-16/18 once six processes had filled the card. `0` turns it off | 0.6 |
| `MIID_GPU_TRACE` | Record the Python allocation stack of every live GPU block and log, at each request start, the largest groups by allocating code and what grew since the previous request (`GPU leak trace` lines) | on |
| `MALLOC_ARENA_MAX` | glibc setting, read at process start: how many malloc arenas the process may create. By default every thread that allocates gets its own 64 MB arena, and a miner has hundreds of threads (torch, OpenBLAS, tokenizers, the axon); 2 keeps the heap in one place. Set it in the process environment (pm2 `env`, `scripts/miner/start_hotkeys.sh` sets it) | glibc default (8 × cores) |
| `MIID_TORCH_THREADS` | CPU threads for this miner process: the torch pool, and (set before anything imports) the `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` defaults, so OpenBLAS and OpenMP stop spawning a thread per advertised host core (128 here against a 15-core cgroup quota; each thread also costs a malloc arena). With several miners on one machine give each its share of the quota | the cgroup quota |
| `MIID_GENERATION_BUDGET_SECONDS` | Upper bound on one request's generation; the request's own deadline usually binds first (see below) | 900 |
| `MIID_DEADLINE_MARGIN_SECONDS` | Safety margin kept before the validator's timeout | 30 |

**Request timing.** Requests from different validators queue for the one GPU and are served
earliest-deadline-first (`MIID/miner/gpu_scheduler.py`); `forward` runs the work in a worker thread
so queued requests are still accepted. Each request's deadline is the validator's send time (its
nonce) plus its 1200 s timeout, minus upload, voice and the margin above. Every slot gets one
attempt first (the request's *first pass*). On FLUX.2 klein the prompts of all slots are encoded
in one text-encoder pass, the transformer and VAE then stay on the GPU for the rest of the request
(`MIID_RESIDENT_GB`), and each take is judged — identity, head turn, scene, edit measurements,
face blend, sibling check — in a worker thread while the next slot generates, so the card is
busy with denoising for nearly the whole first pass. Everything after that runs at a lower GPU
priority: before each attempt the request hands the GPU to any more urgent queued work and queues
again behind it — every first pass goes first, then slots under the identity target, then the
rest, and within a level the earliest deadline wins. With the default `MIID_IDENTITY_RETRIES=0`
that later work is polish only (`MIID_POLISH_ATTEMPTS`): first the slots that missed a target,
re-rolled at full strength with a hint for a measured edit miss, then the satisfied slots short
of `MIID_IDENTITY_GOAL`, re-rolled on their first-pass prompt. With retries on, identity rescues
(the edit held back) and edit retries run before polish. Retries and polish stop when a queued
request needs the GPU to make its own deadline. The GPU is handed back — and the resident weights
evicted — before encryption and upload. A response that would arrive after the timeout scores
nothing, so near the deadline slots are skipped rather than submitted late.

**Face blend.** Only the top 50 miners of a graded round are paid, ranked on validation score with
identity as the tie-break, and most miners now grade 5/5 on most slots, so the 50th place is in
practice decided by identity preservation (0.79–0.93 on the rounds of 2026-10-15). The grader's
identity score is the same AdaFace comparison the miner runs (identical to three decimals on 40
graded slots). FLUX.2 Klein regenerates the whole image, so even a background slot — where the
request says the subject stays unchanged — drifts to 0.68–0.74. `MIID/miner/face_blend.py`
therefore blends the base image's face back into every background take that is short of
`MIID_FACE_BLEND_GOAL`, re-measures the take (identity, scene, accessory) and keeps the blend only
if identity rose without weakening a measured edit. The region is the take's own visible face
skin, found from the inner face outward and limited to where the take still resembles the base
(`MIID_FACE_BLEND_MASK`): a head covering, hat or band worn over the base's hair or forehead is
left exactly as generated, because restoring the base there would erase the requested accessory
while still reading as "a head covering" to the CLIP check. It never pastes: the strongest blend
leaves a fifth of the take's own texture in the face, and a result within 20 dHash bits of the
seed is refused, since the grader scores a copy of the seed -1. Lighting slots are blended with a
*local* colour match (mean and spread in a window of a fifth of an eye distance), so the take's
lighting direction, shadows and cast are carried onto the restored face; the measured lighting
magnitude must stay within 85% of the raw take's and must not drop an intensity level. The window
matters: the grader's lighting delta on these slots is nearly always the left/right asymmetry, a
step at the nose line, and a window of half an eye distance turned it into a ramp (69% of the
magnitude kept, 2 of 18 takes passed); a fifth keeps 88% and passes 17 of 18. Expression slots
restore only the brows, eyes and nose; the mouth, cheeks and chin keep the generated expression,
and the CLIP expression reading must not fall. Pose slots are never blended: the face has turned,
and the landmark fit refuses it. Nor is the lighting+expression combo: the relit, re-expressed
face sits too far from the base, and every mask and match variant lost identity on it (−0.05 to
−0.07 on 10 archived takes). During the first pass each blend runs in a background thread while the next slot
generates, so the GPU does not wait for it; retries and polish blend inline. The archive records
each blend under `variation_report[].blend` (raw and final identity, strength, mask, match, the
lighting or expression measure before and after, or why it was declined). Measured offline on 80 archived
background takes (2026-10-15, strengths 0.6 and 0.8): 71 blended, identity 0.71 → 0.84 on average
(median 0.85), takes at 0.85 or better from 3% to 49%, every accessory verdict unchanged, the
wanted-scene probability down only 0.03 on average, nothing within 93 dHash bits of its seed, about
4 s per take on the CPU (less live, where AdaFace runs on the GPU). Live on the afternoon of
2026-10-15 (four miners, 680 background blends): 0.69 → 0.84 on average. On 29 archived lighting
takes: 24 blended, identity 0.77 → 0.86 (takes at 0.85 or better from 4% to 58%), lighting
magnitude 4.1 → 3.6, three refused for weakening the lighting. On 28 expression takes: 13 blended,
0.66 → 0.71; the rest refused because the fit was poor (the expression moved the mouth corners)
or identity did not rise.

To compare a change against what was actually submitted, replay archived requests (miner
stopped, GPU free) and benchmark the result:

```bash
python -m MIID.miner.replay_requests --out /tmp/replay miner_requests/<date>/<record>...
bash scripts/miner/benchmark.sh /tmp/replay/<date>/*
```

Example with custom configuration:
```bash
export HF_TOKEN="hf_..."
export FLUX_DEVICE="cuda"
export MIID_MODEL="flux_klein"
python neurons/miner.py --netuid 54 --wallet.name your_wallet_name --wallet.hotkey your_hotkey --subtensor.network finney --neuron.logging.debug
```

---

## How It Works

### Face Image Variation (Phase 4)

1. The miner receives an image request containing:
   - A base face image (base64-encoded)
   - A list of variation requests, each specifying a **type** and **intensity**
   - The schema supports multiple variation requests, but current validator runtime sends **one variation per request** and cycles sequentially

2. For each requested variation, the miner:
   - Decodes the base image
   - Generates a variation using the diffusion pipeline (`generate_variations.py`) with the specified parameters
   - Validates that face identity is preserved (AdaFace similarity check)
   - Encrypts the result using drand timelock encryption when available
   - Uploads the encrypted image to S3

3. The miner returns signed S3 submission references to the validator.

### Voice Clone (Phase 5 Cycle 1)

Same round as images when the validator attaches a `voice_request`:

#### Current Approach

1. Validator fetches a reference WAV from the MIID API and draws target speech text (English or Spanish), e.g. `target_text = "three orange six two nine apple table one"` (also sent as `target_words` list).
2. `MIID/miner/voice_generator.generate_voice_clone` synthesizes a WAV with the same speaker identity speaking `target_text`, using Chatterbox Multilingual (zero-shot cloning, English + Spanish). It runs **inside the miner process**, on the same GPU as the image path. Install it into `miner_env` once with `bash scripts/miner/setup_voice.sh --warmup` (see the note on packages below). Without those packages the miner simply returns no voice submission.
3. Up to `MIID_VOICE_ATTEMPTS` (3) takes are generated and the one closest to the reference by SpeechBrain ECAPA similarity (`speechbrain/spkrec-ecapa-voxceleb`, the validator's identity model) is kept. It stops early at `MIID_VOICE_IDENTITY_TARGET` (0.7). A clip below **0.6**, the validator's identity threshold, is dropped.
4. The existing encrypt → S3 → `S3Submission(variation_type="voice")` path runs and fills `voice_s3_submissions`.
5. Voice is collected for **UAV post-grading only** (no live KAV grade for voice).

The models take roughly 3 GB of VRAM while a voice request is being served. Afterwards they are parked in CPU memory (about 3 GB of RAM) so the VRAM is free for the next round's image generation, and moved back in about a second when the next voice request arrives. At startup the miner loads them into CPU memory in the background (`MIID_VOICE_PRELOAD=0` to load on first use instead). Set `MIID_VOICE_KEEP_LOADED=1` to keep them on the GPU between rounds, or `MIID_VOICE_DEVICE=cpu` to keep voice off the GPU entirely (slower; a CUDA out-of-memory also falls back to the CPU on its own). Smoke-test the stack with:

```bash
python -m MIID.miner.voice_generator --reference some_voice.wav --text "three orange six two nine" --language en --out clone.wav
```

**About the packages.** `chatterbox-tts` publishes pins (`torch==2.6`, `transformers==5.2`, `diffusers==0.29`, `numpy<2`) that would downgrade and break the FLUX image stack, so `setup_voice.sh` installs it with `--no-deps` and takes its real runtime dependencies from `requirements-voice.txt`. The combination was verified on torch 2.14 / transformers 5.19 / diffusers 0.41. `pip check` will keep listing those pins as conflicts; that is expected.

#### Updated Approach from `main` branch

1. Validator fetches a reference WAV from the MIID API and draws target speech text (English or Spanish): **5 digit-words + 10 pool words = 15 tokens**, e.g. `target_text = "three orange six book two nine apple table one door light music phone school"` (also sent as `target_words` list). Clip length is not fixed — speak the prompt naturally.
2. Implement `MIID/miner/voice_generator.generate_voice_clone` to synthesize a WAV with the same speaker identity speaking `target_text`.
3. The existing encrypt → S3 → `S3Submission(variation_type="voice")` path runs and fills `voice_s3_submissions`.
4. Voice is collected for **UAV post-grading only** (no live KAV grade for voice).

**Variation types and intensities (Cycle 2):**

| Type | Light | Medium | Far |
|------|-------|--------|-----|
| **Pose edit** | ±15° rotation (slight tilt/turn) | ±30° rotation (clear head turn) | >±45° rotation (near-profile) |
| **Expression edit** | Slight smile, minor brow movement | Smile, serious, mildly surprised | Laughing, surprised, concerned |
| **Lighting edit** | Subtle brightness/contrast change | Directional light, noticeable shadows | Strong shadows, dramatic contrast |
| **Background edit** | Color shift, blur adjustment | Different environment type | Dramatic/contrasting environment |

**Screen replay (Cycle 2 only):** Simulate the face as shown on a device screen (e.g. photo on phone/tablet/laptop/monitor/TV). At least **two** of these visual cues must be clearly visible: moiré/pixel grid, screen glare hotspots, perspective/keystone distortion, gamma or contrast shift typical of display capture, or edge/crop cues (screen borders, bezel reflections).

Validators may also request **accessories** on background edits (head coverings, hats, etc.).

---

## How the Image Models Work

The miner runs `flux_klein` by default; `MIID_MODEL` selects one of the others. The main ones:

| Model | Description | VRAM needed |
|-------|-------------|-------------|
| **FLUX.2-klein** (`flux_klein`) | Fast baseline model; lowest overhead and most stable default path | ~8 GB |
| **PuLID** (`pulid`) | Identity-focused path: tries Nunchaku PuLID on CUDA, otherwise falls back to FLUX.1-Kontext | ~12 GB+ (Nunchaku path); fallback depends on Kontext |

`flux_kontext` (FLUX.1-Kontext-dev, ≥24 GB) and `qwen` (Qwen-Image-Edit-2511) are wired too. There is
no PuLID adapter for the FLUX.2-klein 4B: the published PuLID-FLUX2 weights are sized for the 9B.

If a model fails to load, the miner falls back to `flux_klein`. Licenses, pipeline details, and model wiring are documented in `MIID/miner/generate_variations.py`.

### Recommended alternatives (easy to enable now)

These are already wired in code and can be enabled by setting `MIID_MODEL`:

1. **FLUX.1-Kontext (`flux_kontext`)**
   - Strong text-guided editing quality.
   - Best on higher-memory GPUs (commonly ~24 GB class).
   - Model: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

2. **Qwen Image Edit (`qwen`)**
   - Strong instruction-following image editing.
   - Requires newer diffusers build and `torchvision`.
   - Model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

---

## Reward System

**Validators score miners exclusively on face image variation quality.** There is no separate reward path for name, DOB, or address variations.

You can view your live scores and cycle rewards on the [Yanez Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/).

### How Rewards Are Calculated

Each validation round:
1. Validators send image variation challenges to miners
2. Miners generate and upload encrypted face variations to S3
3. Submissions go through automated pre-checks, identity preservation checks, and manual validation
4. Validators set weights based on image variation scores

**Important:** Phase 5 Cycle 1 Sandbox runs **Sep 24 – Oct 15, 2026** (voice clone for UAV post-grading + continued face image KAV path). Cycle 1 Execution goes live **Oct 15 – Nov 16, 2026**. Phase 5 Cycle 2 Sandbox follows **Nov 16 – Nov 30, 2026**.

### Face Variation Scoring

Face variation rewards are **reputation-based**. Your face variation scores feed into your overall miner reputation, which determines your share of the reward pool.

Face variations go through a multi-stage validation pipeline:

1. **Automatic pre-checks**: Corrupt images, duplicates of the seed, and duplicates of other miners' submissions are filtered out immediately
2. **Identity preservation**: AdaFace similarity must be above threshold (ada_sim >= 0.6 for basic pass)
3. **Variation detection**: Automated systems detect what type of variation was actually produced
4. **Manual validation**: Reviewers check whether the miner followed the query, preserved identity, and produced the exact requested transformation

**Score hints from automated checks:**
- **Score 0**: Failed pre-checks (corrupt, duplicate, identity not preserved)
- **Score -1**: Duplicate detected
- **Score 2**: Partial match (more than one variation detected, or intensity mismatch)
- **Score 3**: Acceptable (identity preserved, label close to claimed)
- **Score 5**: Exact transformation match with identity preserved

For detailed examples of accepted and rejected face variations, see [Face Variation Reward System (PDF)](Face%20Variation%20Reward%20System.pdf).

## How Does a Miner Compete?

Miners compete to generate the highest-quality face image variations:

- Run the image generation pipeline reliably while minimizing latency and compute cost
- Produce **exactly** the requested transformation — no more, no less — while preserving the subject's identity
- Experiment with better diffusion models, fine-tuned parameters, and generation settings
- Validate identity preservation locally before submitting (AdaFace similarity >= 0.7 is the miner-side threshold)
- Build and maintain reputation over time for higher reward tiers

## Common Face Variation Rejection Reasons

- **Identity not preserved**: The generated face looks like a different person (ada_sim below 0.4). Use models and settings that maintain facial features.
- **Copy-paste backgrounds**: Simply pasting the face onto a new background results in negative scores. The background edit must be a natural transformation.
- **Multiple unintended variations**: If asked for a pose edit but the expression also changes significantly, the submission gets a partial match penalty.
- **Corrupted/unreadable images**: Ensure your generation pipeline outputs valid, uncorrupted images.
- **Duplicates**: Submitting the seed image as-is or duplicating another miner's output.

## Performance Tips

1. **Use a GPU** — image generation on CPU is too slow for production and you will not keep up with validator requests.
2. **Tune your generation parameters**: Check `MIID/miner/generate_variations.py`.
   - `NUM_INFERENCE_STEPS` (default 20): Increase for higher quality (slower), decrease for speed.
   - `GUIDANCE_SCALE` (default 3.5): Adjust how closely the model follows the prompt.
3. **Timeout Awareness**: Phase 4 requests have an extended timeout (+15 minutes) to allow for generation, but speed still matters for throughput.
4. **Validate identity preservation locally** before submitting (AdaFace similarity >= 0.7 is the miner-side threshold).
5. **Produce only the requested variation type and intensity** — extra changes hurt your score.
6. **Experiment** with different diffusion models or fine-tuned checkpoints for better results.
7. **Ensure background edits are natural transformations**, not copy-paste composites.
8. Ensure reliable internet connectivity and monitor logs for errors.
9. Keep your miner code up to date with the latest repository changes.

---

## Troubleshooting

### "Phase 4 image generation: DISABLED (missing packages)"

```bash
pip install -r requirements-miner.txt
```

Then complete the AdaFace, Hugging Face token, GPU device, and model download steps above and restart. **Without this enabled, you cannot earn rewards.**

### "Missing Hugging Face token"

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### "Failed to load model — falling back to flux_klein"

Normal if Kontext cannot load.

### AdaFace / checkpoint issues

See [Step 7](#step-7-set-up-adaface-face-identity-validation).

### Port 8091

See [Network Setup Guide](network_setup.md).

### Out of memory

```bash
export MIID_MODEL="flux_klein"
```

---

## Running in the Background

See [Background Mining Guide](background_mining.md). Export `HF_TOKEN` and `FLUX_DEVICE` before starting.

---

## Reference Documents

- [Yanez Dashboard & Leaderboard](https://tao-ui-dashboard.yanez.ai/) — Live miner scores, cycle rewards, and reputation leaderboard
- [Background Mining Guide](background_mining.md) — Run in the background with tmux or pm2
- [Network Setup Guide](network_setup.md) — Port and firewall configuration
- [Logging Guide](logging.md) — Logging setup and management
- [Yanez Identity Generation Bittensor Subnet (PDF)](Yanez%20Identity%20Generation%20Bittensor%20Subnet.pdf) — Full subnet architecture, roadmap, and phase descriptions
- [Face Variation Reward System (PDF)](Face%20Variation%20Reward%20System.pdf) — Face variation validation pipeline, scoring, and examples of accepted/rejected submissions
