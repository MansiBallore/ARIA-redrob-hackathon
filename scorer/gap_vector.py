"""
gap_vector.py — ARIA Technique 5: JD Gap Vector Analysis

Instead of just scoring "how good is this candidate?", we compute
EXACTLY which JD requirements are met and which are gaps.

This serves two purposes:
  1. Better scoring — missing a CORE requirement hurts more than missing
     a nice-to-have. We weight gaps by requirement importance.
  2. Better reasoning — we can generate specific, honest reasoning:
     "Strong on retrieval (Milvus, RAG) but no production deployment signals"
     instead of generic "good candidate with relevant skills"

The JD Gap Vector is a dict of requirement → (met, strength, evidence):
  requirement: string name of the JD requirement
  met: True/False
  strength: 0-1 how well it's met (if met)
  evidence: specific text evidence from the profile
"""

from typing import Dict, Tuple, List


# ── JD Requirements with weights ─────────────────────────────────────────────
# weight: how critical is this requirement to the JD?
# hard_requirement: if True, not meeting this is a major penalty

JD_REQUIREMENTS = {
    "retrieval_or_ranking_systems": {
        "weight": 0.25,
        "hard": True,
        "description": "Built retrieval, ranking, or recommendation systems",
        "skill_signals": [
            "faiss", "pinecone", "qdrant", "milvus", "weaviate", "elasticsearch",
            "rag", "semantic search", "hybrid search", "embedding",
            "recommendation", "ranking", "ltr", "collaborative filtering",
        ],
        "career_signals": [
            "retrieval", "ranking", "recommendation", "vector search",
            "semantic", "search system", "ranked", "rerank",
        ],
    },
    "production_deployment": {
        "weight": 0.20,
        "hard": True,
        "description": "Shipped ML/AI systems to production with real users",
        "skill_signals": [
            "mlops", "model serving", "inference", "triton", "bentoml",
            "sagemaker", "kubeflow",
        ],
        "career_signals": [
            "production", "deployed", "shipped", "real users", "a/b test",
            "rollout", "latency", "throughput", "serving",
        ],
    },
    "python_and_ml_stack": {
        "weight": 0.15,
        "hard": True,
        "description": "Python + core ML libraries (PyTorch, HuggingFace, XGBoost etc.)",
        "skill_signals": [
            "python", "pytorch", "tensorflow", "huggingface", "transformers",
            "xgboost", "lightgbm", "sklearn", "nlp", "fine-tuning", "lora",
        ],
        "career_signals": [
            "python", "pytorch", "tensorflow", "huggingface", "model training",
        ],
    },
    "product_company_experience": {
        "weight": 0.15,
        "hard": False,
        "description": "Worked at product company (not pure IT services)",
        "skill_signals": [],
        "career_signals": [],
        "special": "company_type_check",
    },
    "experience_band": {
        "weight": 0.10,
        "hard": False,
        "description": "5-9 years of experience (ideal band for JD)",
        "skill_signals": [],
        "career_signals": [],
        "special": "yoe_check",
    },
    "evaluation_framework": {
        "weight": 0.08,
        "hard": False,
        "description": "Experience with evaluation: NDCG, A/B testing, offline eval",
        "skill_signals": ["ndcg", "mrr", "map", "offline eval", "online eval"],
        "career_signals": [
            "ndcg", "mrr", "evaluation", "a/b test", "metric", "benchmark",
            "precision@", "recall@",
        ],
    },
    "scale_and_infra": {
        "weight": 0.07,
        "hard": False,
        "description": "Distributed systems, Kafka, Kubernetes, cloud infra",
        "skill_signals": ["kafka", "kubernetes", "spark", "docker", "redis", "aws", "gcp"],
        "career_signals": ["scale", "distributed", "kafka", "kubernetes", "stream"],
    },
    "location_fit": {
        "weight": 0.05,
        "hard": False,
        "description": "Located in or willing to relocate to Pune/Noida/metro",
        "skill_signals": [],
        "career_signals": [],
        "special": "location_check",
    },
    "code_writing": {
        "weight": 0.05,
        "hard": False,
        "description": "Actively writes code (GitHub activity)",
        "skill_signals": [],
        "career_signals": [],
        "special": "github_check",
    },
}

SERVICES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis",
}
METRO_CITIES = ["pune", "noida", "bangalore", "bengaluru", "mumbai",
                "hyderabad", "delhi", "gurgaon", "gurugram", "chennai"]


def _check_requirement(
    req_name: str,
    req_config: dict,
    candidate: dict,
) -> Tuple[bool, float, str]:
    """
    Check a single JD requirement against a candidate.
    Returns: (met, strength 0-1, evidence string)
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    skill_names = {s.get("name", "").lower() for s in skills}
    skill_names_adv = {
        s.get("name", "").lower() for s in skills
        if s.get("proficiency") in ["advanced", "expert"]
    }

    career_text = " ".join(
        f"{j.get('title','')} {j.get('description','')}".lower()
        for j in history
    )

    special = req_config.get("special", "")

    # ── Special checks ─────────────────────────────────────────────────────
    if special == "yoe_check":
        yoe = profile.get("years_of_experience", 0)
        if 5 <= yoe <= 9:
            return True, 1.0, f"{yoe:.1f}yr (ideal band)"
        elif 4 <= yoe < 5:
            return True, 0.75, f"{yoe:.1f}yr (slightly below ideal)"
        elif 9 < yoe <= 12:
            return True, 0.70, f"{yoe:.1f}yr (above ideal band)"
        elif 3 <= yoe < 4:
            return False, 0.40, f"{yoe:.1f}yr (below required)"
        else:
            return False, 0.20, f"{yoe:.1f}yr (significantly outside band)"

    if special == "company_type_check":
        company = profile.get("current_company", "").lower()
        is_services = any(svc in company for svc in SERVICES)
        all_services = all(
            any(svc in j.get("company", "").lower() for svc in SERVICES)
            for j in history
        )
        if all_services:
            return False, 0.10, "Services-only career (JD flags this negatively)"
        elif is_services:
            # Has some product experience
            product_jobs = [
                j for j in history
                if not any(svc in j.get("company","").lower() for svc in SERVICES)
                and j.get("company_size","") in {"11-50","51-200","201-500","501-1000"}
            ]
            if product_jobs:
                co = product_jobs[0].get("company","")
                return True, 0.65, f"Mixed: current at services but has product exp ({co})"
            return False, 0.30, f"Currently at services company"
        else:
            size = profile.get("current_company_size", "")
            if size in {"11-50", "51-200", "201-500", "501-1000"}:
                co = profile.get("current_company", "")
                return True, 1.0, f"Product company: {co} ({size} employees)"
            return True, 0.80, f"Non-services: {profile.get('current_company','')}"

    if special == "location_check":
        location = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        relocate = signals.get("willing_to_relocate", False)
        for city in METRO_CITIES:
            if city in location:
                return True, 1.0, f"Located in {profile.get('location','')}"
        if country == "india" and relocate:
            return True, 0.75, f"India-based, willing to relocate"
        if country == "india":
            return True, 0.55, f"{profile.get('location','')} — not metro, not relocating"
        return False, 0.10, f"Outside India: {profile.get('location','')}"

    if special == "github_check":
        gh = signals.get("github_activity_score", -1)
        if gh == -1:
            return False, 0.30, "No GitHub linked"
        elif gh == 0:
            return False, 0.25, "GitHub linked but inactive"
        elif gh >= 60:
            return True, min(gh / 80, 1.0), f"Active GitHub (score {gh:.0f})"
        elif gh >= 30:
            return True, 0.60, f"Moderate GitHub activity (score {gh:.0f})"
        else:
            return True, 0.35, f"Low GitHub activity (score {gh:.0f})"

    # ── Standard skill + career signal check ──────────────────────────────
    skill_signals = req_config.get("skill_signals", [])
    career_signals = req_config.get("career_signals", [])

    # Find matching skills
    matched_skills = []
    for sig in skill_signals:
        for s_name in skill_names_adv:
            if sig in s_name or s_name in sig:
                matched_skills.append(s_name)
                break

    # Find matching career signals
    matched_career = []
    for sig in career_signals:
        if sig in career_text:
            matched_career.append(sig)

    skill_strength = min(len(matched_skills) / max(len(skill_signals) * 0.3, 1), 1.0)
    career_strength = min(len(matched_career) / max(len(career_signals) * 0.3, 1), 1.0)

    combined_strength = 0.45 * skill_strength + 0.55 * career_strength

    if combined_strength >= 0.60:
        evidence_parts = []
        if matched_skills:
            evidence_parts.append(f"skills: {', '.join(matched_skills[:3])}")
        if matched_career:
            evidence_parts.append(f"career: {', '.join(matched_career[:2])}")
        evidence = "; ".join(evidence_parts) if evidence_parts else "inferred from profile"
        return True, combined_strength, evidence
    elif combined_strength >= 0.25:
        evidence = f"partial: {', '.join(matched_skills[:2] + matched_career[:1])}"
        return False, combined_strength, evidence
    else:
        return False, 0.0, "not found in profile"


def compute_gap_vector(candidate: dict) -> dict:
    """
    Compute the full JD Gap Vector for a candidate.

    Returns:
        gap_score (0-1): how well all JD requirements are met
        requirements: dict of req_name → {met, strength, weight, evidence}
        hard_gaps: list of unmet hard requirements
        soft_gaps: list of unmet soft requirements
        met_requirements: list of met requirements with strength
        gap_summary: human-readable gap summary
    """
    requirements = {}
    hard_gaps = []
    soft_gaps = []
    met_requirements = []

    weighted_score = 0.0
    total_weight = 0.0

    for req_name, req_config in JD_REQUIREMENTS.items():
        met, strength, evidence = _check_requirement(req_name, req_config, candidate)
        weight = req_config["weight"]
        is_hard = req_config.get("hard", False)

        requirements[req_name] = {
            "met": met,
            "strength": round(strength, 3),
            "weight": weight,
            "evidence": evidence,
            "description": req_config["description"],
        }

        # Scoring: met = full credit × strength, unmet = 0 (hard) or partial credit
        if met:
            weighted_score += weight * strength
            met_requirements.append((req_name, strength, evidence))
        else:
            if is_hard:
                hard_gaps.append(req_name)
                # Hard gap: partial credit only if strength > 0
                weighted_score += weight * strength * 0.3
            else:
                soft_gaps.append(req_name)
                weighted_score += weight * strength * 0.5

        total_weight += weight

    # Normalize
    gap_score = weighted_score / total_weight if total_weight > 0 else 0.0

    # Hard gap penalty: missing core requirements = exponential penalty
    if len(hard_gaps) >= 2:
        gap_score *= 0.60
    elif len(hard_gaps) == 1:
        gap_score *= 0.80

    gap_score = max(0.0, min(1.0, gap_score))

    # Generate readable gap summary
    unmet_descriptions = []
    for gap in hard_gaps[:2]:
        unmet_descriptions.append(f"missing: {JD_REQUIREMENTS[gap]['description']}")
    for gap in soft_gaps[:1]:
        unmet_descriptions.append(f"weak: {JD_REQUIREMENTS[gap]['description']}")

    met_descriptions = []
    for req_name, strength, evidence in sorted(met_requirements, key=lambda x: -x[1])[:3]:
        met_descriptions.append(f"{req_name.replace('_', ' ')}: {evidence[:60]}")

    return {
        "gap_score": round(gap_score, 4),
        "requirements": requirements,
        "hard_gaps": hard_gaps,
        "soft_gaps": soft_gaps,
        "met_requirements": [(r, round(s, 2), e) for r, s, e in met_requirements],
        "n_hard_gaps": len(hard_gaps),
        "n_soft_gaps": len(soft_gaps),
        "gap_summary": "; ".join(unmet_descriptions) if unmet_descriptions else "no major gaps",
        "strength_summary": "; ".join(met_descriptions[:2]) if met_descriptions else "",
    }
