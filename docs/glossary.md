# Glossary — Yanez / MIID Subnet 54

Terms you will hit while reading this codebase, running a neuron, or reading validator
result JSON. Each entry names the file that defines it, so you can check the current
behaviour rather than trusting this page.

Only **Phase 4** (face image variations) is live and scored. Phase 1–3 vocabulary
(name variations, addresses, LLM prompts) still appears in older docs and in stub
modules — see [Legacy terms](#12-legacy--deprecated-terms) before you act on it.

---

## 1. Network and neuron vocabulary

| Term | Meaning |
|------|---------|
| **Subnet 54 / SN54** | The mainnet Yanez subnet on the `finney` network. Testnet is `--netuid 322` on network `test`. |
| **Yanez / MIID** | The same project. "Yanez" is the product name; `MIID` is the Python package and the name used in endpoints and env vars. |
| **Neuron** | One running process registered on the subnet — either a miner or a validator. Base classes in [`MIID/base/`](../MIID/base/). |
| **Miner** | Generates identity-preserving face variations and returns signed S3 references. [`neurons/miner.py`](../neurons/miner.py). |
| **Validator** | Issues challenges, waits for the reveal, has them graded, and sets weights. [`neurons/validator.py`](../neurons/validator.py). |
| **Axon** | The miner's inbound server. Validators call it; the miner enforces its whitelist here as well as in `blacklist()`. |
| **Dendrite** | The validator's outbound client. Wrapped by `dendrite_with_retries()` in [`MIID/validator/forward.py`](../MIID/validator/forward.py). |
| **Synapse** | One request/response object on the wire. This subnet has exactly one: `IdentitySynapse`. |
| **`IdentitySynapse`** | Carries an `ImageRequest` validator→miner and `List[S3Submission]` miner→validator. 120 s default timeout. [`MIID/protocol.py`](../MIID/protocol.py). |
| **Metagraph** | The on-chain snapshot of the subnet: UIDs, hotkeys, stake, axon endpoints. |
| **UID** | A neuron's slot number in the metagraph (0…n). Scores, weights and rewards are keyed by UID. |
| **Hotkey / coldkey** | The wallet keypairs. The hotkey signs requests and submissions; every signature in this repo is a hotkey signature. |
| **`vpermit_tao_limit`** | Stake ceiling (default 40960) above which a validator-permitted UID is skipped when sampling miners. [`MIID/utils/uids.py`](../MIID/utils/uids.py). |
| **Weights** | The per-UID emission split the validator publishes on chain, derived from EMA scores. |
| **`spec_version`** | The `version_key` published with weights, derived from `__version__` in [`MIID/__init__.py`](../MIID/__init__.py). Still `"0.0.0"` → spec version `0`. Bump it deliberately when the protocol changes. |
| **EMA / `moving_average_alpha`** | Exponential moving average applied in `update_scores()`, alpha `0.15`. One bad round moves a miner's score slowly. |
| **`--mock`** | Runs either neuron against fake wallets/metagraph from [`MIID/mock.py`](../MIID/mock.py). |
| **Whitelisted validator** | `Miner.WHITELISTED_VALIDATORS` in [`neurons/miner.py`](../neurons/miner.py) — a hardcoded hotkey set, enforced twice. A validator not listed there gets a 401. |

---

## 2. The validator round

| Term | Meaning |
|------|---------|
| **Forward / forward pass** | One complete validation round, ~1 hour of wall clock — not a fast loop. [`MIID/validator/forward.py`](../MIID/validator/forward.py). |
| **Challenge** | The unit of work in one round: one base image plus five requested variations. |
| **`challenge_id`** | Round identifier, `challenge_{unix_ts}_{first 8 chars of validator hotkey}`. It namespaces S3 keys and every signature. |
| **Cycle** | The roadmap label stamped into results, currently `"Phase4-C6-Sandbox"`. It is duplicated in two places in `forward.py` (`phase4_image_data` and `_collect_screen_replay_data`) and must be changed in both. |
| **Sandbox vs execution** | Roadmap phase words. Sandbox = calibration and stability testing with the normal KAV flow; execution = reputation-driven allocation goes live. |
| **`sample_size`** | How many miners a round samples (default 250). |
| **`batch_size`** | How many of those are queried at once (default 150). |
| **Retry rule** | `dendrite_with_retries()` retries **only when more than 50** miners failed. At ≤50 failures the failures get an empty default response instead. Reads backwards at a glance — it is deliberate. |
| **HTTP 422** | Protocol mismatch between a miner and validator running different field sets. Handled explicitly by the retry loop. |
| **Session schedule** | 0–20 min batch 1, 20–40 min batch 2, T+40 min drand unlock, 40–60 min grading window. [`MIID/validator/drand_utils.py`](../MIID/validator/drand_utils.py). |
| **`EPOCH_MIN_TIME`** | 360 s floor on a forward pass, so a fast round still paces itself. |
| **Testnet detection** | Automatic, from the netuid + network + chain endpoint triple. On testnet the wandb project changes, the upload to the MIID server is skipped, and local result JSON is kept instead of deleted. |

---

## 3. Images and seeds

| Term | Meaning |
|------|---------|
| **Base image** | The per-round face the synthetic variations are generated from. Fetched with a signed POST to `$MIID_IMAGES_SERVER/image/<hotkey>` and never written to disk. [`MIID/validator/base_images.py`](../MIID/validator/base_images.py). |
| **IOTD — image of the day** | The fixed daily seed used for the real screen-replay task. The same pair (today + tomorrow) is served to every miner and validator for that UTC day. |
| **Daily seed / tomorrow seed** | `daily_seed_image` and `tomorrow_seed_image` on `ImageRequest`. Tomorrow's is sent a day early so operators can prepare captures before UTC midnight. |
| **Seed pool** | The static image set in [`MIID/validator/fixed_image/`](../MIID/validator/fixed_image/) that ships with the repo. Used in sandbox mode, when the validator sends only the listing and miners pick a seed themselves. |
| **`VALIDATOR_SENDS_SEED_IMAGE`** | Switch in [`MIID/validator/fixed_images.py`](../MIID/validator/fixed_images.py). `True` = validator ships both seed images. `False` = sandbox mode, pool listing only. |
| **Seed slot** | `"today"` or `"tomorrow"` — which IOTD a cached seed or a staged capture belongs to. |
| **`seed_image_name`** | The seed filename with its extension stripped; it becomes a path component in the S3 key. |

---

## 4. Variation vocabulary

Defined in [`MIID/validator/image_variations.py`](../MIID/validator/image_variations.py).

| Term | Meaning |
|------|---------|
| **Variation** | One requested edit of the base image. A `VariationRequest` carries `type`, `intensity`, `description`, `detail`. |
| **Variation type** | `pose_edit`, `lighting_edit`, `expression_edit`, `background_edit`, plus the wire-level `background_in` / `background_out` and `screen_replay`. |
| **`background_in` / `background_out`** | The two background slots actually put on the wire and in S3 keys (indoor / outdoor). Never emit a bare `background` or `background_edit` as a submission type. |
| **Intensity** | `light`, `medium`, `far` — how far the edit should go. On the miner side these map to denoising strengths 0.35 / 0.55 / 0.75. |
| **Combined variation** | Two types in one image, e.g. `lighting_edit+expression_edit`. Each component draws its own intensity. Miners must upload it under the joined `type` string; `+` becomes `_` in the S3 filename. |
| **Standard challenge set** | The exactly 5 synthetic variations `build_standard_challenge_variations()` returns: indoor background, outdoor background, lighting+expression, lighting+pose, pose+expression. `screen_replay` is deliberately *not* in this set. |
| **Accessory** | A weighted-random extra attached to background slots — religious head covering (65%), brim hat, knit hat, bandana, baseball cap, headphones. Appended to the detail as "Additionally, include: …". |
| **Image requirements** | The composition rule appended to every request: passport-style portrait, 3:4, head-and-shoulders, ~1015×1350. |

---

## 5. Real screen-replay vocabulary

A `screen_replay` is a **physical photograph or video of the IOTD displayed on a real
screen** — not a generated image. It travels outside the normal request/response cycle
and is not KAV-graded; validators collect it for manual review.

| Term | Meaning |
|------|---------|
| **Capture** | One physical shooting event, proven with **two distinct files**. |
| **Face close-up** | The primary file (`s3_key` / `image_hash` / `signature`): face dominant, centred, low distortion. Photo or video depending on the variant. |
| **Environment shot / `*_angle2`** | The secondary file (`s3_key_angle2` etc.): a wider still of the whole device in its surroundings. Distortion is fine. Identical hashes across the two files are rejected locally. |
| **`capture_variant`** | How the seed was prepared before capture — three photo tracks (`seed_unchanged`, `seed_smiling`, `seed_eyes_closed`) and three video tracks (`seed_video_blinking`, `seed_video_smiling`, `seed_video_smile_and_blink`). Older `device_camera` / `synthetic_*` names are aliased to these. |
| **`ScreenReplayUAV`** | The miner-reported metadata block attached to a screen-replay submission: seed filename, UTC date, camera used, device photographed, capture variant, and the cue checklist. |
| **Visual cues** | The five booleans always reported: `moire_pixel_grid`, `screen_glare_hotspots`, `perspective_keystone_distortion`, `gamma_contrast_shift`, `edge_crop_cues`. A genuine capture may show none, some, or all. |
| **`device_photographed`** | The screen the seed was shown on: `phone`, `tablet`, `laptop`, `monitor`, `tv`. Distinct from `camera_used`, which is the device that took the shot. |
| **Cross-view consistency** | The review criterion: both files must look like the same capture — same seed face on screen, same bezel geometry, same lighting/glare direction, distinct hashes. |
| **Due / held** | A capture keyed to tomorrow's IOTD is held by `_screen_replay_is_due()` until that UTC date arrives. |
| **inbox / staged / queue** | The operator workflow in [`MIID/miner/real_image_miner_guide/`](../MIID/miner/real_image_miner_guide/): drop files in `inbox/`, `submit_real_photo.py` moves them to `staged/` and writes `screen_replay.json` with `"ready": true`; extra captures pile up in `queue/`. The running miner picks it up on the next validator request — no restart. |
| **`seeds/`** | Where the miner writes each incoming IOTD pair (plus `seeds.json`) so the operator can display them. |
| **`screen_replay_data`** | The results-JSON block that records what was asked, which IOTD was the seed, and who uploaded. |

---

## 6. Scoring vocabulary

Defined in [`MIID/validator/reward.py`](../MIID/validator/reward.py).

| Term | Meaning |
|------|---------|
| **KAV** | Online image-quality score from the external grading API. Weight `--neuron.kav_weight`, default **0.10** of the miner share. |
| **UAV** | Reputation score carried in the MIID server's snapshot. Weight `--neuron.uav_weight`, default **0.90** of the miner share. |
| *(acronym note)* | The code comments expand these as "Known / Unknown Address Variation"; the roadmap PDFs expand UAV as "Unknown Attack Vectors". Treat KAV as *this round's graded quality* and UAV as *accumulated reputation* — that is what the code does regardless of the expansion. |
| **`validation_score`** | Per-image grade from the API, raw 0–5, normalised by ÷5 into 0–1. |
| **`identity_preservation`** | Per-image 0–1 measure of whether the face is still the same person. Used as the quality gate and as the ranking tiebreaker. |
| **`sampled_variation`** | The grading API usually grades **one** randomly sampled variation slot per round, not all five. When it does, `num_graded = 1` and only that submission is scored. |
| **`quality_threshold`** | Default 0.6. Despite the name it is the **identity-preservation minimum** a miner must clear to earn anything. |
| **`top_miner_cap`** | Default 50. Only the top 50 by composite score are even considered. |
| **Blended ranking** | Qualified miners are re-ranked, then `reward = blend_factor · exp(-decay_rate · rank) + (1 - blend_factor) · raw_score`, with `blend_factor` 0.7 and `decay_rate` 0.05. |
| **Burn event** | Nobody clears the cap + threshold filters → 100% of emissions burn for that round. |
| **`BURN_UID`** | Hardcoded UID **59**, the burn address. |
| **`burn_fraction`** | Default 0.30. |
| **`PARTNER_HOTKEY` / `PARTNER_FRACTION`** | The vetted commercial partner pool, 0.35. If the hotkey is not in the metagraph the fraction reverts to burn (total burn 65%). |
| **Dual incentive** | The standing split: ~35% miners (KAV+UAV), ~30% burn, ~35% partner. The partner pool was funded by cutting burn from 65% → 30%, not by cutting miners. |
| **`burn_mode`** | Which of the eight KAV/UAV/partner-availability branches fired this round (`configured`, `no_uav`, `no_kav_no_partner`, …). Unused KAV weight always burns; unused UAV weight reroutes to the partner when present, otherwise burns. |
| **`rep_score`** | Raw reputation from the server. Decays over time; at ≤0 a miner earns zero UAV regardless of tier. |
| **`rep_tier`** | `Black`, `Platinum`, `Diamond`, `Gold`, `Silver`, `Bronze`, `Neutral`, `Watch` — plus `New` for miners absent from the snapshot. |
| **`TIER_MULTIPLIERS`** | 1.35 down to 0.90 by tier, applied on top of the normalised score. |
| **`normalize_rep_score()`** | Continuous piecewise map from raw `rep_score` to 0.0–4.0. No clamping to tier boundaries — decay shows up proportionally. |
| **New miner** | Not in the reputation snapshot → zero UAV, KAV only, until validated UAVs build reputation. |
| **Unqueried miner** | Not sampled for this round's KAV challenge but present in the snapshot → KAV 0, UAV as normal, still listed so the server applies its decay. |
| **`rep_cache` / `rep_snapshot_version`** | Returned by the server on a successful upload and stored in module globals for the **next** round. UAV reputation is therefore always one round stale, and empty on a validator's first pass. |
| **Reward allocation** | The per-round `{timestamp, rep_snapshot_version, miners}` record sent to the server. Failed uploads queue in `pending_allocations.json` and ride along with the next round. |

Two invariants that are easy to break:

- Burn is applied **exactly once** — either in `get_image_variation_rewards(skip_burn=False)`
  when UAV grading is off, or in `apply_reputation_rewards()` when it is on. Never both.
- `apply_reputation_rewards()` returns extra entries (unqueried miners, burn, partner), so
  its `rewards`/`uids` are **not** positionally aligned with `miner_uids`. Always map by UID.

---

## 7. Submission and storage vocabulary

| Term | Meaning |
|------|---------|
| **`S3Submission`** | What a miner returns: S3 key, media hash, signature, variation type, path signature — never image bytes. |
| **`image_hash`** | SHA-256 of the **original, unencrypted** media. |
| **`path_signature`** | `sign(f"{challenge_id}:{hotkey}")` hex, first 16 chars. Namespaces the miner's S3 prefix so one miner cannot write into another's path. [`MIID/miner/submission_builder.py`](../MIID/miner/submission_builder.py). |
| **`signature`** (per file) | `sign(f"challenge:{challenge_id}:hash:{image_hash}")`, proving the miner produced that exact file. |
| **S3 key** | `submissions/{challenge_id}/{hotkey}/{path_signature}/{seed_image_name}/{variation_type}_{timestamp}.png.tlock`. |
| **`.tlock`** | The timelock-encrypted payload extension. |
| **Timelock encryption** | Drand-based encryption that cannot be opened before a target round. Uses `bittensor.timelock` (from `bittensor_drand`) — *not* a separate `timelock` package. Unavailable → the miner uploads raw bytes with a sandbox-only warning. |
| **drand / Quicknet** | The public randomness beacon providing the reveal. 3-second rounds. [`MIID/validator/drand_utils.py`](../MIID/validator/drand_utils.py). |
| **`target_drand_round` / `reveal_timestamp`** | When decryption becomes possible. Fixed at T+40 min (`reveal_delay_seconds`, default 2400) from challenge start. |
| **Reveal** | The moment the drand round publishes and the grading API can decrypt. `wait_until_reveal()` blocks the validator until then. |
| **S3 upload** | A plain HTTP PUT to a public-write bucket (`yanez-miid-sn54`). No boto3, no AWS credentials. [`MIID/miner/s3_upload.py`](../MIID/miner/s3_upload.py). |
| **Local storage mode** | `MIID_USE_S3=false` writes submissions under `MIID_LOCAL_STORAGE` instead. Used by the dry run. |
| **AdaFace** | The identity-similarity model the miner uses to self-check each variation before submitting. Floor `min_similarity=0.4`. [`MIID/miner/ada_face_compare.py`](../MIID/miner/ada_face_compare.py). |
| **Dry run** | `python -m MIID.miner.dry_run_submission --image face.png` — runs decode → generate → AdaFace → encrypt → upload offline and prints why each variation was kept or dropped. |
| **`PHASE4_AVAILABLE`** | Whether the image stack imported. A miner without it still registers and serves, returns `[]`, and scores 0. |

---

## 8. External services

Hardcoded unless the row says otherwise.

| Name | What it is |
|------|-----------|
| **`MIID_SERVER`** | `http://52.44.186.20:5000/upload_data` — receives the signed results JSON and returns `rep_cache` + `rep_snapshot_version`. Hardcoded in `forward.py`. |
| **`MIID_IMAGES_SERVER`** | Same host by default, but this one *is* an env var. `/image/<hotkey>` serves the per-round base image, `/fixed_image/<hotkey>` the IOTD pair. |
| **`GRADING_API_BASE`** | `http://98.90.28.118:5000` — `/grade_v2` acknowledges the job, `/grade_results` is polled on a 15 min → 5 min → 1 min schedule with a 60 min cap. |
| **The Flask server** | Lives in this repo at [`MIID/datasets/app.py`](../MIID/datasets/app.py) but is deployed separately. |
| **S3 bucket** | `yanez-miid-sn54`, region `us-east-1`. |
| **`miner_api/`** | A standalone operator tool for the Yanez `is_live` / `is_ai` endpoints. Not part of the neuron runtime; it has its own hardcoded wallet names. |

---

## 9. Files and state on disk

| Path | What it holds |
|------|---------------|
| `validator_results/` | Per-round result JSON. Deleted after a successful upload on mainnet; kept on testnet. |
| `validator_results/phase4_state.json` | `global_index`, the cycling position, advanced once per forward pass. |
| `validator_results/pending_allocations.json` | Reward allocations whose upload failed, re-sent with the next round. |
| `MIID/validator/fixed_image_cache/` | Cached IOTD pair plus `seed_meta.json` (filenames and UTC dates per slot). |
| `MIID/validator/fixed_image/` | The static sandbox seed pool shipped with the repo. |
| `MIID/miner/real_image_miner_guide/` | `inbox/`, `staged/`, `queue/`, `seeds/`, `screen_replay.json`. |
| `miner_requests/` (repo root) | **Request archive** — one `record.json` per validator approach: the request, the submissions returned, the per-variation drop reasons and the model used. Written on every outcome, including refusals and crashes. Gitignored. [`MIID/miner/request_archive.py`](../MIID/miner/request_archive.py). |
| `vali.env` | Validator environment file, read from the **current working directory**, not the repo root. |
| `miner_env/` · `validator_env/` | The venvs the setup scripts create at the repo root. |

---

## 10. Environment variables and notable flags

| Name | Effect |
|------|--------|
| `HF_TOKEN` | Hugging Face token; required to pull the diffusion weights (miner only). |
| `FLUX_DEVICE` | `cuda`, `mps`, or `cpu`. |
| `MIID_MODEL` | Forces one model instead of the random pick. |
| `MIID_MODEL_RANDOM` | Defaults to `1` — random base model per query. |
| `MIID_INFERENCE_STEPS` / `MIID_GUIDANCE_SCALE` | Diffusion knobs, default 20 and 3.5. |
| `MIID_USE_S3`, `MIID_S3_BUCKET`, `MIID_S3_REGION`, `MIID_LOCAL_STORAGE` | Storage routing. |
| `MIID_IMAGES_SERVER` | Base image / IOTD server, default `http://52.44.186.20:5000` (validator only). The only external endpoint that is overridable. |
| `MIID_REQUEST_ARCHIVE`, `MIID_ARCHIVE_ENABLED`, `MIID_ARCHIVE_IMAGES`, `MIID_ARCHIVE_MAX_RECORDS`, `MIID_ARCHIVE_RETENTION_HOURS` | Request-archive location, on/off, whether media is kept (off by default), and retention. |
| `BT_NO_PARSE_CLI_ARGS` | Set to `false` by [`MIID/utils/config.py`](../MIID/utils/config.py). bittensor ≥ 10.5 defaults it to `true`, which silently drops every CLI arg and leaves `config.neuron` missing. Do not remove it. |
| **Flags you cannot turn off** | `--neuron.UAV_grading`, `--wandb.disable`, `--wandb.cleanup_runs`, `--neuron.nominatim_cache_enabled` are `action="store_true"` with `default=True`. Changing them means editing `config.py`. |

---

## 11. Model vocabulary

Defined in [`MIID/miner/generate_variations.py`](../MIID/miner/generate_variations.py). One of the
base models is picked at random per query unless `MIID_MODEL` forces one; a load failure falls
back to `flux_klein`.

| Key | What it is |
|-----|-----------|
| `flux_klein` | FLUX.2-klein-4B. The lightweight default and the fallback. |
| `pulid` | PuLID ("Pure and Lightning ID Customization") via Nunchaku on CUDA, falling back to FLUX.1-Kontext. Highest identity fidelity. |
| `pulid_flux2` | FLUX.2 Klein backbone with PuLID-FLUX2 adapter weights. |
| `flux_kontext` | FLUX.1-Kontext-dev. Wired but opt-in; needs ≥24 GB VRAM. |
| `qwen` | Qwen-Image-Edit-2511. Wired but opt-in. |
| **Base models** | The three entries with `base: True` — the random-selection pool. |
| **Nunchaku** | The quantised inference runtime PuLID uses when available on CUDA. |

---

## 12. Legacy / deprecated terms

These appear in older docs, PDFs and stub files. They describe removed systems — do not
build on them.

| Term | Status |
|------|--------|
| **Phase 1–3** | Name, DOB, address and location variation tasks. Removed. |
| **`query_generator` / `rule_evaluator` / `rule_extractor` / `cheat_detection`** | One-line stubs kept only so old imports resolve. |
| **Ollama / LLM name variations** | Gone. Validators run no local model; miners run only diffusion. |
| **YEVS** | Referenced as "YEVS-style" in the variation type definitions. Naming lineage only. |
| **LDS V1** | Phase 3 post-validation system named in the README roadmap and a PDF. |
| **Nominatim** | Geocoding cache from the location phase. The config flags survive; nothing uses them. |
| **`uav_data`** | The Phase 3 results block whose shape `screen_replay_data` now mirrors. |
| **`tests/` and `unittest/`** | Phase 1–3 leftovers importing symbols that no longer exist. `pytest tests/` fails at collection. There is no working test suite. |
| **`docs/test.md`** | A corrupted dump describing the removed Ollama pipeline. Ignore it. |
