"""How long each validator really waits for a response.

Every request says it times out after synapse.timeout (1200 s), but some
validators stop waiting well before that. A response they no longer wait for
is never graded, which is worse than a response with fewer retries behind it.
Measured by matching archived requests to their feed grades, 2026-11-13..17:

- Yuma graded 9 of 10 responses sent back within 640 s, and 2 of 87 slower ones.
- Rizzo graded most responses up to ~850 s, and 3 of 19 after 880 s.
- Yanez and RoundTable21 graded at the same rate up to 1200 s.

MIID_VALIDATOR_BUDGETS lists "<validator>:<seconds>" pairs, comma-separated.
<validator> is a name from the miner's whitelist (Miner.WHITELISTED_VALIDATORS)
or a hotkey prefix. An empty value switches the limits off. The seconds count
from the validator's send time, like its timeout, and the miner's upload
reserve and margin come out of them.
"""
import os
from typing import Dict, Mapping, Optional

DEFAULT_BUDGETS = "Yuma:600,Rizzo:800"


def parse_budgets(spec: str) -> Dict[str, float]:
    """{validator name or hotkey prefix: seconds} from a MIID_VALIDATOR_BUDGETS value."""
    budgets: Dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, seconds = item.rpartition(":")
        try:
            value = float(seconds)
        except ValueError:
            value = 0.0
        if not sep or not key.strip() or value <= 0:
            raise ValueError(
                f"MIID_VALIDATOR_BUDGETS: bad entry {item!r} (expected <validator>:<seconds>)"
            )
        budgets[key.strip()] = value
    return budgets


BUDGETS = parse_budgets(os.environ.get("MIID_VALIDATOR_BUDGETS", DEFAULT_BUDGETS))


def budget_for(
    hotkey: Optional[str],
    names: Mapping[str, str],
    budgets: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    """Seconds this validator waits for a response, or None when only its timeout applies.

    ``names`` maps validator hotkeys to their names (the miner's whitelist).
    """
    budgets = BUDGETS if budgets is None else budgets
    if not hotkey:
        return None
    name = names.get(hotkey)
    if name is not None and name in budgets:
        return budgets[name]
    for key, seconds in budgets.items():
        if hotkey.startswith(key):
            return seconds
    return None
