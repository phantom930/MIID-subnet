"""Per-validator response budgets (MIID/miner/response_budget.py, 2026-11-17).

Run with miner_env: ``python -m pytest unittest/test_miner_response_budget.py``
"""
import pytest

from MIID.miner import response_budget as rb

NAMES = {
    "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs": "Yuma",
    "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54": "Rizzo",
    "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr": "Yanez",
}


def test_default_limits_yuma_and_rizzo_only():
    budgets = rb.parse_budgets(rb.DEFAULT_BUDGETS)
    assert rb.budget_for("5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs", NAMES, budgets) == 600
    assert rb.budget_for("5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54", NAMES, budgets) == 800
    assert rb.budget_for("5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr", NAMES, budgets) is None
    assert rb.budget_for(None, NAMES, budgets) is None


def test_hotkey_prefix_matches_validators_outside_the_whitelist():
    budgets = rb.parse_budgets("5Dvgtk1b:900")
    assert rb.budget_for("5Dvgtk1bqLycAyyc2VFKvAqmpvoQf5oWsD4qnu6vqbdWSL54", {}, budgets) == 900
    assert rb.budget_for("5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr", {}, budgets) is None


def test_empty_value_switches_the_limits_off():
    assert rb.parse_budgets("") == {}
    assert rb.parse_budgets(" , ") == {}


@pytest.mark.parametrize("spec", ["Yuma", "Yuma:", ":600", "Yuma:soon", "Yuma:0", "Yuma:-5"])
def test_bad_entries_are_refused(spec):
    with pytest.raises(ValueError):
        rb.parse_budgets(spec)
