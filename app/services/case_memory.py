"""
Case-based memory for the RL agent -- answers "have we seen a state like this before, and
how did it actually turn out" and, when a new case's outcome disagrees with what similar past
cases suggested, "what specifically was different this time." This sits ALONGSIDE
LinearQPolicy (rl_engine.py), not instead of it: the Q-policy already generalizes across
states via its learned weights, but a linear function can't tell you WHICH past experience it
"remembers" behind any given number, or flag when a state looks unusually close to (or
unlike) history. This module makes that explicit and inspectable.

Every RLSignal since state got added to its schema carries the exact RL_FEATURE_NAMES-ordered
vector it was decided from (see main.py's create_rl_signal). Once that signal resolves
(hit/miss/expired -- never superseded, see below), it becomes a "case": a (state, real
outcome) pair this module can compare a brand new state against.

Deliberately NOT a similarity search library (no faiss/annoy/sklearn KDTree dependency) --
the candidate pool for one pair/interval is at most a few hundred to a few thousand resolved
signals, small enough that a plain O(n) distance scan is instant and keeps this dependency-
free and easy to read line by line.
"""
import math
from typing import Optional

# superseded is deliberately excluded from the candidate pool everywhere in this module --
# same reasoning as RLSignal.status's own docstring and GET /rl/accuracy's
# directional_hit_rate_pct: a superseded signal never actually resolved (the agent changed
# its mind before label_outcome got the chance), so it isn't real evidence of what happens
# when a state like this plays out. Feeding it in as a "case" would mean remembering "the
# agent flip-flopped here" as if it were a market outcome, which it isn't.
RESOLVED_STATUSES = ("hit", "miss", "expired")

# How many nearest historical cases to summarize by default -- small enough that the
# summary stays about genuinely similar states (a huge k would start pulling in cases that
# aren't really comparable), large enough that one freak outcome doesn't dominate the
# hit-rate estimate. Not independently tuned -- same starting-guess caveat as every other
# unvalidated constant in this project; revisit once there's enough case history to check
# different k values against each other.
DEFAULT_K = 10


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Plain L2 distance -- state features are already roughly comparable in scale by
    construction (votes in [-1,1], the raw_* features normalized similarly, see
    rl_engine.RL_MARKET_FEATURE_NAMES's own comment), so no extra standardization step is
    needed before comparing distances directly."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_cases(
    target_state: list[float], candidates: list[dict], k: int = DEFAULT_K,
) -> list[dict]:
    """
    candidates: dicts with at least "state" (list[float], same length as target_state) --
    any extra keys (e.g. "signal_id", "status", "outcome_pct_move") are carried through
    unchanged onto the returned entries, with "distance" added. Candidates missing "state"
    or with a mismatched length (a legacy pre-state-field signal, or a signal from a since-
    changed feature schema) are silently skipped, same "don't fail, just skip what isn't
    usable" convention as score_pending_rl_signals' skipped_no_data.

    Returns up to k entries, nearest first.
    """
    scored = []
    for c in candidates:
        state = c.get("state")
        if not state or len(state) != len(target_state):
            continue
        scored.append({**c, "distance": euclidean_distance(target_state, state)})
    scored.sort(key=lambda c: c["distance"])
    return scored[:k]


def memory_summary(target_state: list[float], candidates: list[dict], k: int = DEFAULT_K) -> dict:
    """
    The main entry point: "here's what happened, historically, in situations like this one."

    hit_rate_pct is computed over hit+miss neighbors only (excludes expired from the
    denominator) -- same directional-accuracy reasoning as GET /rl/accuracy's
    directional_hit_rate_pct added this session: an "expired" neighbor is a non-event, not a
    loss, and folding it into the denominator would make a pattern that mostly just times out
    look like a "usually loses" pattern instead of the more accurate "usually doesn't decide"
    one. expired_pct is still reported separately so that distinction stays visible.
    """
    nearest = nearest_cases(target_state, candidates, k)
    if not nearest:
        return {
            "cases_found": 0, "hit_rate_pct": None, "avg_pct_move": None,
            "expired_pct": None, "avg_distance": None, "nearest": [],
        }

    hits = sum(1 for c in nearest if c.get("status") == "hit")
    misses = sum(1 for c in nearest if c.get("status") == "miss")
    expired = sum(1 for c in nearest if c.get("status") == "expired")
    decided = hits + misses
    pct_moves = [c["outcome_pct_move"] for c in nearest if c.get("outcome_pct_move") is not None]

    return {
        "cases_found": len(nearest),
        "hit_rate_pct": round(hits / decided * 100, 1) if decided else None,
        "avg_pct_move": round(sum(pct_moves) / len(pct_moves), 4) if pct_moves else None,
        "expired_pct": round(expired / len(nearest) * 100, 1),
        "avg_distance": round(sum(c["distance"] for c in nearest) / len(nearest), 4),
        "nearest": [
            {k2: c[k2] for k2 in ("signal_id", "status", "outcome_pct_move", "distance") if k2 in c}
            for c in nearest
        ],
    }


def explain_divergence(
    target_state: list[float], neighbor_state: list[float], feature_names: list[str], top_n: int = 5,
) -> list[dict]:
    """
    "This case's closest historical neighbor had a different outcome -- what actually
    changed?" Returns the top_n features (by absolute difference) between the two states,
    largest first, each as {"feature": name, "this_case": value, "neighbor_case": value,
    "diff": signed difference}. Not a claim of causation (a linear-distance-ranked feature
    list isn't a causal attribution method) -- it's the same "make the reasoning
    inspectable, don't just trust a bare number" principle as this project's SignalReason.
    detail on every rule-based signal and feature_coefficients on the ML classifier: a human
    reviewing a surprising outcome gets a concrete starting point ("volume_ratio was way
    higher this time") instead of an opaque "the model was wrong."
    """
    diffs = [
        {"feature": name, "this_case": round(t, 4), "neighbor_case": round(n, 4), "diff": round(t - n, 4)}
        for name, t, n in zip(feature_names, target_state, neighbor_state)
    ]
    diffs.sort(key=lambda d: abs(d["diff"]), reverse=True)
    return diffs[:top_n]


# Below this many similar cases, memory doesn't know enough to override the Q-policy at all
# -- a handful of neighbors is too easily dominated by one or two outliers. Starting guesses,
# not independently tuned -- same caveat as every other unvalidated constant in this project;
# revisit once enough gated/ungated live signals exist to compare.
MIN_CASES_FOR_GATING = 5
# Below this historical hit rate among similar past cases, memory says "don't take this trade
# at all" -- roughly "clearly losing," not just "below breakeven."
BLOCK_HIT_RATE_FLOOR = 25.0
# Below this (but at/above BLOCK_HIT_RATE_FLOOR) -- still below the ~40% breakeven this
# project's fixed 1.5:1 target:stop needs (see rl_engine.py's own comments) -- memory doesn't
# block the trade outright, just refuses to let it size up to LARGE.
DOWNSIZE_HIT_RATE_FLOOR = 40.0
# A policy whose own overall resolved-trade record is at/above this is demonstrably doing
# something right -- a thin, possibly-noisy slice of locally-similar cases (a handful of
# neighbors) shouldn't be enough to fully block a trade from a policy with this track record,
# even though it's still enough to justify the softer downsize-only caution above. Requires
# MIN_DECIDED_FOR_POLICY_STRENGTH real decided trades before trusting this number at all --
# a policy that's gone 3-for-4 isn't "demonstrably strong," it's a tiny sample.
STRONG_POLICY_HIT_RATE_FLOOR = 50.0
MIN_DECIDED_FOR_POLICY_STRENGTH = 10


def memory_gate(memory: dict, size_tier: str) -> tuple[str, Optional[str]]:
    """
    The "act on it, not just report it" half of case memory -- takes the Q-policy's chosen
    size_tier and either leaves it alone, downgrades LARGE to SMALL, or overrides the whole
    trade to "HOLD", based on how similar past states (already filtered to the SAME direction
    by the caller, see main.py's create_rl_signal) actually turned out. Same layered-safety-
    rule philosophy as RL_TARGET_ATR_MULT/RL_STOP_ATR_MULT's own 1.5:1 floor -- "guaranteed by
    construction here, not left for the agent to discover through reward alone" -- applied to
    a different risk dimension (this decision's evidence base, not the reward shape itself).

    memory is expected to carry policy_hit_rate_pct/policy_decided_trades (added by the caller
    in main.py, NOT by memory_summary -- that function is scoped to case-level lookup only,
    this is policy-level context) alongside the usual case-level fields. This exists because a
    policy that's genuinely performing well overall shouldn't be fully blocked just because
    ONE narrow, possibly-noisy local neighborhood looks weak -- a few similar-looking losses
    next to a policy's otherwise-solid track record is exactly the kind of local variance a
    small k-nearest-neighbor sample will show sometimes, not necessarily a real warning sign.
    The softer downsize-to-SMALL still applies regardless of overall strength -- even a good
    policy can have specific setups worth sizing down on -- only the outright HOLD block backs
    off when the policy's own real record says otherwise.

    Returns (new_size_tier_or_"HOLD", reason_or_None). reason is None exactly when no override
    happened (not enough cases, or hit rate wasn't low enough to act on), so the caller can
    tell "gate ran and approved this trade" apart from "gate didn't have an opinion."
    """
    cases_found = memory.get("cases_found", 0)
    hit_rate_pct = memory.get("hit_rate_pct")
    if cases_found < MIN_CASES_FOR_GATING or hit_rate_pct is None:
        return size_tier, None

    policy_hit_rate_pct = memory.get("policy_hit_rate_pct")
    policy_decided_trades = memory.get("policy_decided_trades") or 0
    policy_is_strong = (
        policy_hit_rate_pct is not None
        and policy_decided_trades >= MIN_DECIDED_FOR_POLICY_STRENGTH
        and policy_hit_rate_pct >= STRONG_POLICY_HIT_RATE_FLOOR
    )

    if hit_rate_pct < BLOCK_HIT_RATE_FLOOR:
        if policy_is_strong:
            return "SMALL", (
                f"memory: {cases_found} similar past cases resolved to only {hit_rate_pct}% "
                f"hit rate, which would normally block this trade -- but this policy's own "
                f"overall record is {policy_hit_rate_pct}% over {policy_decided_trades} decided "
                f"trades, so capping size at SMALL instead of blocking outright"
            )
        return "HOLD", (
            f"memory: {cases_found} similar past cases resolved to only {hit_rate_pct}% "
            f"hit rate -- blocking this trade"
        )
    if hit_rate_pct < DOWNSIZE_HIT_RATE_FLOOR and size_tier == "LARGE":
        return "SMALL", (
            f"memory: {cases_found} similar past cases resolved to {hit_rate_pct}% hit rate "
            f"(below breakeven for this project's 1.5:1 target:stop) -- capping size at SMALL"
        )
    return size_tier, None


def find_diverging_neighbor(
    target_status: str, nearest: list[dict],
) -> Optional[dict]:
    """
    From an already-computed nearest-cases list (see nearest_cases), finds the closest
    neighbor whose status DISAGREES with target_status's real-world sense of "good" vs "bad"
    (hit vs miss -- "expired" isn't a disagreement with either, it's just a non-event, so it's
    skipped as a comparison point here). None if no such neighbor exists in the given list
    (e.g. every neighbor was expired, or every neighbor agreed).
    """
    opposite = "miss" if target_status == "hit" else "hit" if target_status == "miss" else None
    if opposite is None:
        return None
    for c in nearest:
        if c.get("status") == opposite:
            return c
    return None
