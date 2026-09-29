# Miner ↔ Validator Workflow

How one validation round actually runs, end to end. Diagrams are Mermaid, so they render
on GitHub. Terms are defined in the [Glossary](glossary.md).

The important thing to hold onto: **one `forward()` is a ~1 hour cycle, not a quick loop**,
and **raw images never cross the wire in either direction** — the miner returns signed S3
references to timelock-encrypted files that nobody can open until T+40 min.

---

## 1. One full round

```mermaid
sequenceDiagram
    autonumber
    participant V as Validator
    participant Y as Yanez API<br/>MIID_IMAGES_SERVER
    participant D as drand Quicknet
    participant M as Miner
    participant S3 as S3 bucket<br/>yanez-miid-sn54
    participant G as Grading API
    participant SRV as MIID server<br/>upload_data
    participant BT as Bittensor chain

    Note over V: T+0 — round starts

    V->>Y: signed POST /fixed_image/hotkey
    Y-->>V: today + tomorrow IOTD, cached daily
    V->>Y: signed POST /image/hotkey
    Y-->>V: base face image, base64, never written to disk
    V->>D: GET /info
    D-->>V: genesis + period, target round at T+40 min
    Note over V: build 5 synthetic variations<br/>challenge_id = challenge_ts_hotkey8

    rect rgb(238, 245, 255)
    Note over V,M: T+0 to T+40 — query in batches of 150, sample size 250
    V->>M: IdentitySynapse with ImageRequest
    M->>M: verify whitelist, else 401
    M->>M: persist IOTD pair to seeds/
    M->>M: generate, AdaFace check, timelock encrypt
    M->>S3: HTTP PUT encrypted .tlock files
    S3-->>M: ok
    M-->>V: List of S3Submission<br/>keys, hashes, signatures only
    end

    Note over V,D: T+40 — wait_until_reveal blocks here
    V->>D: poll until target round published
    D-->>V: round signature available

    rect rgb(240, 248, 240)
    Note over V,G: T+40 to T+60 — grading window
    V->>G: signed POST /grade_v2 with phase4_image_data
    G-->>V: 200 processing, ack only
    G->>S3: download .tlock files
    G->>D: fetch round signature
    G->>G: decrypt and score
    loop 15 min, then 5 min, then every 1 min, 60 min cap
        V->>G: POST /grade_results
        G-->>V: processing, or completed with result
    end
    end

    V->>V: KAV ranking, then UAV from last round's snapshot, then burn
    V->>BT: set_weights with version_key = spec_version
    V->>SRV: signed POST results JSON
    SRV-->>V: rep_cache + rep_snapshot_version
    Note over V: stored in module globals<br/>for the NEXT round
```

### Timeline

| Window | What happens |
|--------|--------------|
| T+0 | Sample miners, fetch base image and IOTD pair, compute the drand target round, build the challenge. |
| T+0 – T+20 | Batch 1 queried (up to `batch_size`, default 150). |
| T+20 – T+40 | Batch 2 queried. |
| T+40 | Drand round publishes. Submissions become decryptable. |
| T+40 – T+60 | Grading API decrypts and scores, validator polls for results. |
| End | Weights set, results signed and uploaded, reputation snapshot cached for next round. |

Miners that answer fast do **not** shorten the round — the unlock is pinned to T+40 from
challenge start regardless.

---

## 2. What the validator sends and what comes back

```mermaid
flowchart LR
    subgraph REQ["ImageRequest — validator to miner"]
        direction TB
        R1["base_image — per-round face, base64"]
        R2["variation_requests — exactly 5"]
        R3["daily_seed_image — today's IOTD"]
        R4["tomorrow_seed_image — sent a day early"]
        R5["target_drand_round + reveal_timestamp"]
        R6["challenge_id"]
        R7["real_screen_replay_instructions"]
    end

    subgraph RES["List of S3Submission — miner to validator"]
        direction TB
        S1["s3_key — path to the .tlock file"]
        S2["image_hash — SHA-256 of the ORIGINAL bytes"]
        S3["signature — proves this miner made that file"]
        S4["variation_type"]
        S5["path_signature — namespaces the S3 prefix"]
        S6["s3_key_angle2 and screen_replay_uav<br/>screen_replay only"]
    end

    REQ -->|"IdentitySynapse"| RES
```

The five requested variations are always the same shape, built by
`build_standard_challenge_variations()`:

1. `background_in` — indoor, plus a weighted-random accessory
2. `background_out` — outdoor, plus a weighted-random accessory
3. `lighting_edit+expression_edit`
4. `lighting_edit+pose_edit`
5. `pose_edit+expression_edit`

`screen_replay` is deliberately **not** in this set — see [section 5](#5-the-screen-replay-side-channel).

On the timeout: `IdentitySynapse` declares `timeout: float = 120.0`, but the validator
overrides it per round with `--neuron.timeout` (default **1200 s**) when it constructs the
synapse. A miner sizing its pipeline should plan against 1200 s per batch, not 120.

---

## 3. Inside the miner

Two access checks run before any work starts, then the pipeline in
[`MIID/miner/submission_builder.py`](../MIID/miner/submission_builder.py) runs once per
variation. A variation that fails any step is dropped, not faked — the miner submits
fewer items rather than bad ones.

```mermaid
flowchart TD
    A["Validator request hits the axon"] --> B{"Hotkey in<br/>WHITELISTED_VALIDATORS?"}
    B -->|no| B1["401 NotVerified"]
    B -->|yes| C{"default_verify<br/>signature ok?"}
    C -->|no| B1
    C -->|yes| D{"image_request<br/>present?"}
    D -->|no| D1["return empty list"]
    D -->|yes| E["persist today + tomorrow IOTD<br/>to seeds/ and seeds.json"]
    E --> F{"PHASE4_AVAILABLE?"}
    F -->|no| D1
    F -->|yes| G["decode_base_image"]
    G --> H["generate_variations<br/>random model unless MIID_MODEL set"]

    H --> I["per variation"]
    I --> J{"valid image bytes?"}
    J -->|no| X["drop — corrupt bytes"]
    J -->|yes| K{"AdaFace similarity<br/>at or above 0.4?"}
    K -->|no| X2["drop — identity not preserved"]
    K -->|yes| L["sign_media_hash<br/>challenge:cid:hash:h"]
    L --> M{"timelock available?"}
    M -->|yes| N["encrypt_image_for_drand<br/>to target_round"]
    M -->|no| N2["raw bytes — SANDBOX ONLY, logs a warning"]
    N --> O["HTTP PUT to S3"]
    N2 --> O
    O -->|"no key returned"| X3["drop — upload failed"]
    O -->|"key returned"| P["append S3Submission"]

    P --> Q["attach a due screen_replay<br/>capture if one is staged"]
    Q --> R["return List of S3Submission"]
```

The S3 key is structured so one miner cannot write into another's prefix:

```
submissions/{challenge_id}/{hotkey}/{path_signature}/{seed_image_name}/{variation_type}_{ts}.png.tlock
                                     ^^^^^^^^^^^^^^^^
                                     sign("{challenge_id}:{hotkey}") hex, first 16 chars
```

Run the identical pipeline offline with
`python -m MIID.miner.dry_run_submission --image face.png` — it prints per-variation
outcomes and exits non-zero when nothing was produced.

---

## 4. How the response becomes a weight

```mermaid
flowchart TD
    A["s3_submissions_by_miner"] --> B["POST /grade_v2, then poll /grade_results"]
    B --> C["per-image validation_score 0-5<br/>and identity_preservation 0-1"]
    C --> D["usually ONE sampled variation slot<br/>is graded, not all five"]
    D --> E["composite = avg validation score<br/>+ tiny identity tiebreak"]
    E --> F{"in top 50<br/>AND identity at or above 0.6?"}
    F -->|no| G["reward 0"]
    F -->|yes| H["blended rank curve<br/>0.7 * exp of -0.05 * rank + 0.3 * raw"]
    G --> I["KAV scores"]
    H --> I

    J["rep_cache from the PREVIOUS round"] --> K["normalize_rep_score to 0-4<br/>times TIER_MULTIPLIERS"]
    K --> L["UAV scores<br/>including miners not queried this round"]

    I -->|"weight 0.10"| M["apply_reputation_rewards"]
    L -->|"weight 0.90"| M
    M --> N["rescale each side to its exact target<br/>KAV 0.035, UAV 0.315"]
    N --> O["append BURN_UID 59 at 0.30<br/>and PARTNER_HOTKEY at 0.35"]
    O --> P["update_scores — EMA alpha 0.15"]
    P --> Q["set_weights"]
```

Three things that trip people up here:

- **Burn is applied exactly once.** Either inside `get_image_variation_rewards(skip_burn=False)`
  when UAV grading is off, or inside `apply_reputation_rewards()` when it is on. Never both.
- **The returned arrays are not positionally aligned with `miner_uids`.**
  `apply_reputation_rewards()` adds unqueried miners, the burn UID and the partner UID.
  Always map by UID.
- **UAV reputation is always one round stale.** `rep_cache` arrives in the *upload response*
  at the end of a round and is used by the *next* one, so it is empty on a validator's first pass.

---

## 5. The screen-replay side channel

A `screen_replay` is a physical photograph or video of the IOTD shown on a real screen. It
is **not** generated, **not** part of the 5-variation set, and **not** KAV-graded — validators
collect it for manual review. It also does not follow the request/response rhythm: the
operator stages a capture whenever they have one, and the running miner attaches it to the
next validator request that comes in.

```mermaid
sequenceDiagram
    autonumber
    participant V as Validator
    participant M as Miner process
    participant OP as Operator
    participant FS as real_image_miner_guide/

    V->>M: ImageRequest with today + tomorrow IOTD
    M->>FS: write seeds/ + seeds.json
    Note over OP,FS: out of band, no restart needed
    OP->>OP: display seed on a real screen<br/>capture with a different camera
    OP->>FS: drop 2 files into inbox/
    OP->>FS: run submit_real_photo.py
    FS->>FS: move to staged/, write screen_replay.json ready=true<br/>extra captures go to queue/

    V->>M: next ImageRequest
    M->>FS: pick a DUE capture<br/>tomorrow's seed stays held until that UTC date
    M-->>V: submission with face close-up in primary fields<br/>+ environment shot in *_angle2
    V->>V: collect into screen_replay_data for manual review
```

Every submission bundles **two distinct files** of the same capture — a face close-up
(photo or video, per `capture_variant`) and a wider environment still. Identical hashes are
rejected locally. Reviewers check cross-view consistency: same seed face on screen, same
bezel geometry, same lighting and glare direction, distinct hashes.

---

## 6. Failure paths worth knowing

| Situation | What happens |
|-----------|--------------|
| Miner hotkey not whitelisted on the validator's side | Request returns 401. The miner scores 0 for the round. |
| Miner has no image stack (`PHASE4_AVAILABLE` false) | It still registers and serves, returns `[]`, scores 0. |
| Protocol field mismatch between the two sides | HTTP 422, handled explicitly by `dendrite_with_retries()`. |
| ≤50 miners failed to respond | **No retry.** They get empty default responses. Retries fire only when *more* than 50 failed — the condition reads backwards at a glance but is deliberate. |
| Grading API never returns | Every miner gets 0.0 KAV for the round. |
| Nobody clears top-50 + the 0.6 identity floor | Burn event — 100% of the round's emissions burn. |
| `PARTNER_HOTKEY` absent from the metagraph | Its 35% is added to burn, so burn reverts to 65%. Miner share is unaffected. |
| Results upload fails | The allocation queues in `validator_results/pending_allocations.json` and is re-sent with the next round's payload. Local result files are deleted only after a successful upload, and only on mainnet. |

---

## Source map

| Step | File |
|------|------|
| Round orchestration | [`MIID/validator/forward.py`](../MIID/validator/forward.py) |
| Wire format | [`MIID/protocol.py`](../MIID/protocol.py) |
| Challenge construction | [`MIID/validator/image_variations.py`](../MIID/validator/image_variations.py) |
| Reveal timing | [`MIID/validator/drand_utils.py`](../MIID/validator/drand_utils.py) |
| Scoring and allocation | [`MIID/validator/reward.py`](../MIID/validator/reward.py) |
| Miner request handling | [`neurons/miner.py`](../neurons/miner.py) |
| Miner pipeline | [`MIID/miner/submission_builder.py`](../MIID/miner/submission_builder.py) |
| Upload and key layout | [`MIID/miner/s3_upload.py`](../MIID/miner/s3_upload.py) |
