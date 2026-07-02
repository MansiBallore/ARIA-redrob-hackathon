"""
reasoning.py — ARIA Reasoning Generator v2

Generates specific, fact-grounded 1-2 sentence reasoning per candidate
using the Gap Vector output. Every sentence cites actual profile data.

Submission spec checks:
  - Specific facts from profile (not generic praise)
  - JD connection (why they fit or don't)
  - Honest gaps (rank 80 shouldn't sound like rank 1)
  - No hallucination
  - Variation across candidates
"""

from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 21)


def _days_since(date_str):
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 999


def generate_reasoning(candidate: dict, all_scores: dict, rank: int) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    gap = all_scores.get("gap_vector", {})
    dna = all_scores.get("dna", {})
    velocity = all_scores.get("velocity", {})
    peer = all_scores.get("peer_context", {})

    days_inactive = velocity.get("days_inactive", 999)
    notice_days = velocity.get("notice_days", 60)
    open_to_work = velocity.get("open_to_work", False)
    salary = velocity.get("salary_range_lpa", "unknown")

    dna_seq = dna.get("dna_sequence", "")
    hard_gaps = gap.get("hard_gaps", [])
    met_reqs = gap.get("met_requirements", [])
    gap_summary = gap.get("gap_summary", "")

    # ── Sentence 1: Key strength ───────────────────────────────────────────
    s1_parts = []

    if rank <= 10:
        # Lead with the strongest match evidence
        if met_reqs:
            top_req, top_strength, top_evidence = met_reqs[0]
            s1_parts.append(
                f"{yoe:.1f}yr {title} at {company} — "
                f"strong {top_req.replace('_',' ')} ({top_evidence[:70]})"
            )
        else:
            s1_parts.append(f"{yoe:.1f}yr {title} at {company} with broad ML engineering background")

        if dna.get("has_retrieval"):
            s1_parts.append(f"; DNA includes retrieval experience (sequence: {dna_seq})")

    elif rank <= 30:
        if met_reqs:
            top_req, top_strength, top_evidence = met_reqs[0]
            s1_parts.append(
                f"{yoe:.1f}yr {title} ({peer.get('company_tier','')}) — "
                f"{top_req.replace('_',' ')}: {top_evidence[:60]}"
            )
        else:
            s1_parts.append(f"{yoe:.1f}yr {title} at {company}, {location}")

    elif rank <= 60:
        s1_parts.append(
            f"{yoe:.1f}yr {title} at {company} "
            f"({peer.get('peer_group','').replace('_',' ')} cohort)"
        )
        if met_reqs:
            req, strength, ev = met_reqs[0]
            s1_parts.append(f"; partial fit on {req.replace('_',' ')}: {ev[:50]}")

    else:
        s1_parts.append(
            f"{yoe:.1f}yr {title}; adjacent profile in top-100 based on "
            f"partial overlap with JD requirements"
        )

    sentence1 = "".join(s1_parts) + "."

    # ── Sentence 2: Gaps + signals ─────────────────────────────────────────
    s2_parts = []

    # Availability signals first
    if days_inactive < 7:
        s2_parts.append(f"Active {days_inactive}d ago")
    elif days_inactive < 30:
        s2_parts.append(f"Recently active ({days_inactive}d)")
    elif days_inactive > 120:
        s2_parts.append(f"Inactive {days_inactive}d — availability risk")

    if open_to_work:
        s2_parts.append("open-to-work")

    if notice_days <= 15:
        s2_parts.append(f"immediate joiner")
    elif notice_days > 90:
        s2_parts.append(f"{notice_days}d notice")

    # Gaps
    if hard_gaps and rank > 10:
        gap_names = [g.replace("_", " ") for g in hard_gaps[:2]]
        s2_parts.append(f"gaps: {', '.join(gap_names)}")

    # Salary
    if salary != "unknown" and salary != "0-0":
        s2_parts.append(f"expects {salary} LPA")

    if s2_parts:
        sentence2 = "; ".join(s2_parts) + "."
    else:
        sentence2 = f"Located {location}; response rate {signals.get('recruiter_response_rate', 0):.0%}."

    return f"{sentence1} {sentence2}"
