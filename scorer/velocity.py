"""
velocity.py — ARIA Technique 3: Velocity-Weighted Signals

Standard signal scoring treats all signals as static averages.
But hiring urgency is about NOW — not what someone's average was.

Key insight: Two candidates can have identical average signal scores but
very different "momentum":
  - Candidate A: active yesterday, applied last week, saved by 5 recruiters
  - Candidate B: same averages, but all activity was 8 months ago

Candidate A has VELOCITY. They're actively in the market RIGHT NOW.

We apply exponential time-decay to every temporal signal and compute
a "velocity score" that captures current market momentum.

Additionally, we use the salary range to check budget fit — a candidate
expecting 80 LPA for a role budgeted at 30-40 LPA wastes everyone's time.
"""

import math
from datetime import date, datetime


REFERENCE_DATE = date(2026, 6, 21)

# Estimated budget for Senior AI Engineer at Redrob (inferred from JD)
JD_BUDGET_MIN_LPA = 25.0
JD_BUDGET_MAX_LPA = 55.0


def _days_since(date_str: str) -> int:
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 999


def _exp_decay(days: int, half_life: int) -> float:
    """Exponential decay: score = e^(-ln2 * days / half_life)"""
    return math.exp(-0.693 * days / half_life)


def velocity_score(candidate: dict) -> dict:
    """
    Compute velocity score using time-decayed behavioral signals.

    Unlike static signal averaging, this captures CURRENT momentum.

    Components:
    1. Activity velocity      — how recently and actively are they on platform?
    2. Market demand velocity — are recruiters looking at them NOW?
    3. Application momentum   — are they actively pursuing opportunities?
    4. Reliability signal     — response rate + interview completion (static but important)
    5. Budget alignment       — salary expectations vs JD budget

    Returns dict with velocity_score and detailed breakdown.
    """
    signals = candidate.get("redrob_signals", {})

    # ── 1. Activity Velocity ──────────────────────────────────────────────────
    days_inactive = _days_since(signals.get("last_active_date", ""))
    signup_days = _days_since(signals.get("signup_date", ""))

    # Recency decay: half-life = 30 days (very aggressive — 60 days = 25% score)
    activity_recency = _exp_decay(days_inactive, half_life=30)

    # Platform tenure bonus (longer on platform = more invested)
    tenure_bonus = min(signup_days / 365, 1.0) * 0.10

    # Open to work: hard multiplier
    otw_multiplier = 1.0 if signals.get("open_to_work_flag") else 0.55

    activity_velocity = min(1.0, (activity_recency + tenure_bonus) * otw_multiplier)

    # ── 2. Market Demand Velocity ─────────────────────────────────────────────
    # These are inherently "last 30 days" metrics — already time-windowed
    views_30d = signals.get("profile_views_received_30d", 0)
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    appearances_30d = signals.get("search_appearance_30d", 0)

    # Normalize: 50 views = good, 15 saves = excellent, 300 appearances = excellent
    demand_velocity = (
        0.40 * min(views_30d / 50, 1.0) +
        0.40 * min(saved_30d / 10, 1.0) +
        0.20 * min(appearances_30d / 300, 1.0)
    )

    # ── 3. Application Momentum ───────────────────────────────────────────────
    apps_30d = signals.get("applications_submitted_30d", 0)
    # 1-2 apps = targeted (good), 3-5 = active, 6+ = desperate (slight penalty)
    if apps_30d == 0:
        app_score = 0.20   # passive candidate
    elif apps_30d <= 2:
        app_score = 0.80   # targeted, quality applications
    elif apps_30d <= 5:
        app_score = 1.00   # actively looking
    elif apps_30d <= 8:
        app_score = 0.85   # very active, slightly spray-and-pray
    else:
        app_score = 0.65   # spray-and-pray mode

    # ── 4. Reliability (static but weighted) ─────────────────────────────────
    response_rate = signals.get("recruiter_response_rate", 0.3)
    avg_response_hrs = signals.get("avg_response_time_hours", 48)
    interview_rate = signals.get("interview_completion_rate", 0.7)
    offer_rate = signals.get("offer_acceptance_rate", -1)

    # Response time scoring
    if avg_response_hrs <= 6:
        time_score = 1.0
    elif avg_response_hrs <= 24:
        time_score = 0.85
    elif avg_response_hrs <= 72:
        time_score = 0.60
    else:
        time_score = max(0.10, 1.0 - avg_response_hrs / 300)

    offer_score = 0.65 if offer_rate == -1 else max(0.2, min(1.0, offer_rate))

    reliability = (
        0.35 * response_rate +
        0.30 * interview_rate +
        0.20 * time_score +
        0.15 * offer_score
    )

    # ── 5. Budget Alignment ───────────────────────────────────────────────────
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max", 0)

    if sal_min == 0 and sal_max == 0:
        budget_score = 0.70  # unknown = neutral
    else:
        # Check overlap between candidate expectation and JD budget
        overlap_min = max(sal_min, JD_BUDGET_MIN_LPA)
        overlap_max = min(sal_max, JD_BUDGET_MAX_LPA)

        if overlap_min <= overlap_max:
            # There's overlap — score based on how well aligned
            overlap_size = overlap_max - overlap_min
            candidate_range = max(sal_max - sal_min, 1)
            budget_score = min(1.0, 0.6 + 0.4 * (overlap_size / candidate_range))
        else:
            # No overlap — candidate expects more or less
            if sal_min > JD_BUDGET_MAX_LPA:
                # Expects too much — significant mismatch
                gap = sal_min - JD_BUDGET_MAX_LPA
                budget_score = max(0.1, 1.0 - gap / 30)
            else:
                # Expects less — fine, might accept
                budget_score = 0.85

    # ── Notice Period (with decay for long periods) ──────────────────────────
    notice_days = signals.get("notice_period_days", 60)
    if notice_days <= 0:
        notice_score = 1.0
    elif notice_days <= 15:
        notice_score = 0.95
    elif notice_days <= 30:
        notice_score = 0.85
    elif notice_days <= 60:
        notice_score = 0.65
    elif notice_days <= 90:
        notice_score = 0.45
    else:
        notice_score = max(0.05, 1.0 - notice_days / 200)

    # ── GitHub Activity (code-writing signal) ────────────────────────────────
    gh = signals.get("github_activity_score", -1)
    if gh == -1:
        gh_score = 0.35   # no github = mild negative for ML engineer
    elif gh == 0:
        gh_score = 0.30
    else:
        gh_score = min(gh / 65, 1.0)

    # ── Verification ─────────────────────────────────────────────────────────
    verify_score = (
        (0.5 if signals.get("verified_email") else 0.0) +
        (0.3 if signals.get("verified_phone") else 0.0) +
        (0.2 if signals.get("linkedin_connected") else 0.0)
    )

    # ── Combine all components ────────────────────────────────────────────────
    combined = (
        0.28 * activity_velocity +
        0.20 * reliability +
        0.15 * demand_velocity +
        0.12 * app_score +
        0.10 * budget_score +
        0.08 * notice_score +
        0.07 * gh_score +
        0.00 * verify_score   # verification is a must-have check, not weighted
    )

    # Hard floor: unverified email = major credibility penalty
    if not signals.get("verified_email"):
        combined *= 0.70

    combined = max(0.0, min(1.0, combined))

    return {
        "velocity_score": round(combined, 4),
        "activity_velocity": round(activity_velocity, 3),
        "demand_velocity": round(demand_velocity, 3),
        "reliability": round(reliability, 3),
        "app_momentum": round(app_score, 3),
        "budget_aligned": budget_score >= 0.60,
        "budget_score": round(budget_score, 3),
        "notice_score": round(notice_score, 3),
        "github_score": round(gh_score, 3),
        "days_inactive": days_inactive,
        "notice_days": notice_days,
        "open_to_work": bool(signals.get("open_to_work_flag")),
        "salary_range_lpa": f"{sal_min}-{sal_max}" if sal_max > 0 else "unknown",
    }
