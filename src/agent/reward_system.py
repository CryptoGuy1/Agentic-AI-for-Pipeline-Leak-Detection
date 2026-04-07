# =========================================================
# reward_system.py  —  FIXED VERSION
# =========================================================
#
# BUG FIXED:
#   The old version used ANOMALY THRESHOLDS only.
#   Mixture gas has anomaly ≈ 0.10 (lowest of all gases).
#   So the old reward said "Monitor" is correct for Mixture,
#   which is backwards — Mixture needs Emergency Shutdown.
#
# FIX:
#   Use gas_id + action, identical to the training notebook.
#   This is the ONLY correct reward signal for this dataset.
#
# Action semantics (same as training notebook):
#   0 = Monitor           → correct for NoGas   (gas_id=0)
#   1 = Increase Sampling → correct for Perfume  (gas_id=3, low anomaly)
#   2 = Req Verification  → correct for Perfume  (gas_id=3, high anomaly)
#   3 = Raise Alarm       → correct for Smoke    (gas_id=1)
#   4 = Emergency         → correct for Mixture  (gas_id=2)
#
# gas_id mapping (from GAS_MAP in training notebook):
#   NoGas=0, Smoke=1, Mixture=2, Perfume=3
# =========================================================

CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
DANGER_GAS_IDS  = {1, 2}   # Smoke and Mixture are hazardous


def compute_reward(
    state,
    action: int,
    gas_id: int = None,
    anomaly: float = None,
) -> float:
    """
    Compute reward for the agent's action.

    ALWAYS pass gas_id whenever you have ground-truth labels
    (verify mode, training, evaluation).

    Only fall back to anomaly-only in true blind live mode
    where no label exists at all — and even then, treat the
    result as an approximation only.

    Args:
        state   : state vector (used to read anomaly if anomaly arg not given)
        action  : int 0-4
        gas_id  : int (0=NoGas, 1=Smoke, 2=Mixture, 3=Perfume)
                  Pass this whenever available — it is always correct.
        anomaly : float override. If None, reads state[0].

    Returns:
        float reward
    """
    norm_anomaly = float(anomaly) if anomaly is not None else float(state[0])

    if gas_id is not None:
        return _reward_gas_id(int(gas_id), int(action), norm_anomaly)
    else:
        # WARNING: anomaly-based reward is unreliable for this dataset.
        # Mixture (most dangerous) has the LOWEST anomaly score.
        # Only use this path in truly blind live mode.
        return _reward_anomaly_only(norm_anomaly, int(action))


def _reward_gas_id(gas_id: int, action: int, anomaly: float) -> float:
    """
    Exact replica of training notebook reward function.
    Uses gas_id + action — always correct.
    Reward range: approximately [-2.33, +1.0] after /3.0 scaling.
    """
    reward = 0.0
    correct = CORRECT_ACTIONS.get(gas_id, [])

    # ── Base correctness ──────────────────────────────────────
    if action in correct:
        reward += 2.0
        if gas_id in DANGER_GAS_IDS:
            reward += anomaly           # severity bonus (0 to 1)
    else:
        reward -= 2.0

    # ── Safety-critical penalties ──────────────────────────────
    if gas_id in DANGER_GAS_IDS and action == 0:
        reward -= 10.0   # WORST: missed dangerous gas → Monitor

    if gas_id == 0 and action >= 3:
        reward -= 4.0    # False emergency alarm on clean air

    # ── Precision bonuses ──────────────────────────────────────
    if gas_id == 1 and action == 3:    # Smoke → Raise Alarm ✓
        reward += 1.0
    if gas_id == 2 and action == 4:    # Mixture → Emergency ✓
        reward += 1.0

    # ── Perfume severity calibration ───────────────────────────
    if gas_id == 3:
        if anomaly > 0.5 and action == 2:    # High-anomaly Perfume → Req Verification ✓
            reward += 0.8
        elif anomaly > 0.5 and action == 1:  # Under-responding to strong Perfume
            reward -= 0.5
        elif anomaly <= 0.5 and action == 1: # Low-anomaly Perfume → Increase Sampling ✓
            reward += 0.5

    # ── Scale (wider than old /6.0 to avoid Q-value saturation) ─
    return float(reward / 3.0)


def _reward_anomaly_only(anomaly: float, action: int) -> float:
    """
    Fallback for truly blind live mode (no ground-truth label at all).
    WARNING: anomaly is NOT a reliable danger proxy for this dataset.
    Mixture (most dangerous) has anomaly ≈ 0.10 — lower than NoGas!
    Do NOT use this for evaluation or training.
    """
    if anomaly < 0.20:
        rewards = {0: 2.0, 1: 0.5, 2: -0.25, 3: -2.0, 4: -3.0}
    elif anomaly < 0.50:
        rewards = {0: 0.25, 1: 1.5, 2: 1.25, 3: -1.0, 4: -2.0}
    elif anomaly < 0.80:
        rewards = {0: -3.0, 1: 0.75, 2: 2.5, 3: 1.5, 4: -0.5}
    elif anomaly < 0.90:
        rewards = {0: -6.0, 1: -2.0, 2: 0.75, 3: 4.0, 4: 2.5}
    else:
        rewards = {0: -8.0, 1: -4.0, 2: -0.5, 3: 3.5, 4: 6.0}
    return float(rewards.get(action, 0.0))


def get_expected_action(gas_id: int) -> list:
    """Return the list of correct actions for a given gas_id."""
    return CORRECT_ACTIONS.get(int(gas_id), [])


def is_correct_action(gas_id: int, action: int) -> bool:
    """Return True if action is correct for this gas_id."""
    return int(action) in CORRECT_ACTIONS.get(int(gas_id), [])