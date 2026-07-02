"""
peer_benchmark.py — ARIA Technique 4: Peer Benchmark Percentile

Most ranking systems score candidates on an absolute scale.
Problem: a 7yr ML engineer at a Series B startup is fundamentally
different from a 7yr ML engineer at Infosys — but absolute scoring
can't capture this contextual difference.

Solution: Define peer groups, score each candidate relative to their
cohort, then use the PERCENTILE within the peer group as a signal.

This gives us "best in their cohort" candidates, not just "above average."

Peer Groups (defined by YOE band × company tier):
  YOE bands: junior (0-4), mid (5-9), senior (10+)
  Company tiers: product (startup/tech), hybrid, services

A candidate in the mid/product group who ranks in the 90th percentile
of that group is more valuable than an average-scoring mid/services candidate
even if their absolute raw scores are similar.
"""

from typing import List, Dict, Tuple


# ── Company tier classification ──────────────────────────────────────────────

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree", "l&t infotech",
}

PRODUCT_COMPANIES_SIGNALS = [
    # Industry signals
    "saas", "ai/ml", "fintech", "edtech", "healthtech", "e-commerce",
    "startup", "series a", "series b", "series c",
    # Size signals (small-mid product companies)
]

PRODUCT_COMPANY_SIZES = {"11-50", "51-200", "201-500", "501-1000"}


def _company_tier(candidate: dict) -> str:
    """
    Classify candidate's company as 'product', 'hybrid', or 'services'.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])

    company = profile.get("current_company", "").lower()
    industry = profile.get("current_industry", "").lower()
    size = profile.get("current_company_size", "")

    # Check current company first
    for svc in SERVICES_COMPANIES:
        if svc in company:
            # Check if they also have product experience in history
            product_count = sum(
                1 for j in history
                if j.get("company_size", "") in PRODUCT_COMPANY_SIZES
                and not any(svc in j.get("company", "").lower() for svc in SERVICES_COMPANIES)
            )
            if product_count >= 1:
                return "hybrid"
            return "services"

    # Check if current is a product company
    if size in PRODUCT_COMPANY_SIZES:
        for signal in PRODUCT_COMPANIES_SIGNALS:
            if signal in industry or signal in company:
                return "product"
        return "product"  # small company without explicit signals = still product

    return "hybrid"  # default


def _yoe_band(yoe: float) -> str:
    if yoe < 4:
        return "junior"
    elif yoe <= 9:
        return "mid"
    else:
        return "senior"


def assign_peer_group(candidate: dict) -> str:
    """
    Assign a peer group label to a candidate.
    Format: "mid_product", "senior_services", etc.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    yoe_band = _yoe_band(yoe)
    co_tier = _company_tier(candidate)
    return f"{yoe_band}_{co_tier}"


def compute_peer_percentiles(
    candidates: list,
    raw_scores: List[float],
) -> List[float]:
    """
    For each candidate, compute their percentile rank within their peer group.

    Args:
        candidates: list of candidate dicts
        raw_scores: list of raw fit scores (same order as candidates)

    Returns:
        List of percentile scores (0-1) for each candidate.
    """
    # Group candidates by peer group
    peer_groups: Dict[str, List[Tuple[int, float]]] = {}

    for i, (c, score) in enumerate(zip(candidates, raw_scores)):
        group = assign_peer_group(c)
        if group not in peer_groups:
            peer_groups[group] = []
        peer_groups[group].append((i, score))

    # Compute percentile within each group
    percentiles = [0.0] * len(candidates)

    for group, members in peer_groups.items():
        n = len(members)
        if n == 1:
            # Only member of their group — assign 0.5 (neutral)
            idx, _ = members[0]
            percentiles[idx] = 0.50
            continue

        # Sort by score
        sorted_members = sorted(members, key=lambda x: x[1])

        for rank, (idx, score) in enumerate(sorted_members):
            # Percentile: rank / (n-1), so worst = 0.0, best = 1.0
            percentile = rank / (n - 1) if n > 1 else 0.5
            percentiles[idx] = percentile

    return percentiles


def peer_context(candidate: dict) -> dict:
    """Return peer group context for a candidate."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    return {
        "peer_group": assign_peer_group(candidate),
        "yoe_band": _yoe_band(yoe),
        "company_tier": _company_tier(candidate),
    }
