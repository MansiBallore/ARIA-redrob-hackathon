"""
skill_graph.py — ARIA Technique 2: Implicit Skill Graph

Instead of checking "does candidate have skill X?", we measure how DEEP
they are inside the retrieval/ML skill cluster.

Concept: Build a graph where skills are nodes, and edges connect skills
that commonly co-occur in careers. A candidate who has skills that are
tightly connected to each other AND to the JD core skills is a deeper
fit than someone with isolated skills.

Implementation (no external graph library needed):
  - Pre-define the JD skill cluster as a set of interconnected groups
  - Score each candidate by how many cluster nodes they activate
  - Give bonus for activating adjacent nodes (skill neighborhood density)
  - Validate with endorsements + assessment scores (no free rides)

Cluster structure (derived from the JD and domain knowledge):
  Core → Adjacent → Extended → Irrelevant
"""

import math
from typing import Dict, List


# ── Skill Cluster Graph ────────────────────────────────────────────────────────
# Each group has a "center" (most JD-relevant) and "neighbors"
# Activating the center = high value; neighbors = partial value

SKILL_CLUSTERS = {
    "retrieval_core": {
        "value": 1.00,
        "skills": {
            # Vector databases
            "faiss": 1.0, "pinecone": 1.0, "qdrant": 1.0, "milvus": 1.0,
            "weaviate": 1.0, "chroma": 0.9, "pgvector": 0.8,
            "opensearch": 0.85, "elasticsearch": 0.85,
            # Retrieval methods
            "rag": 1.0, "hybrid search": 1.0, "semantic search": 0.95,
            "dense retrieval": 0.95, "sparse retrieval": 0.90,
            "bi-encoder": 0.90, "cross-encoder": 0.90, "reranking": 0.90,
            # Embeddings
            "embedding": 0.95, "sentence-transformer": 0.95,
            "text embedding": 0.90, "bge": 0.85, "e5": 0.85,
            # Evaluation
            "ndcg": 0.90, "mrr": 0.85, "map": 0.80,
        }
    },
    "ranking_recommendation": {
        "value": 0.90,
        "skills": {
            "learning to rank": 1.0, "ltr": 1.0,
            "recommendation system": 0.95, "recommender": 0.95,
            "collaborative filtering": 0.90, "content-based filtering": 0.85,
            "matrix factorization": 0.85, "bandit": 0.80,
            "ranking model": 0.90, "pointwise": 0.80,
            "listwise": 0.80, "pairwise": 0.80,
        }
    },
    "ml_engineering": {
        "value": 0.85,
        "skills": {
            "pytorch": 0.95, "tensorflow": 0.90, "huggingface": 0.95,
            "transformers": 0.95, "fine-tuning": 0.90, "lora": 0.85,
            "qlora": 0.85, "peft": 0.85, "llm": 0.90,
            "nlp": 0.90, "bert": 0.85, "gpt": 0.85,
            "xgboost": 0.80, "lightgbm": 0.80, "catboost": 0.75,
            "sklearn": 0.80, "scikit-learn": 0.80,
            "mlflow": 0.80, "wandb": 0.80, "weights & biases": 0.80,
        }
    },
    "production_ml": {
        "value": 0.80,
        "skills": {
            "model serving": 0.90, "inference": 0.85, "triton": 0.85,
            "onnx": 0.80, "torchserve": 0.80, "ray serve": 0.80,
            "bentoml": 0.75, "mlops": 0.85, "feature store": 0.80,
            "feast": 0.75, "kubeflow": 0.80, "sagemaker": 0.75,
        }
    },
    "infra_scale": {
        "value": 0.65,
        "skills": {
            "kafka": 0.85, "spark": 0.80, "kubernetes": 0.80,
            "docker": 0.75, "redis": 0.75, "airflow": 0.75,
            "celery": 0.70, "rabbitmq": 0.70, "gcp": 0.70,
            "aws": 0.70, "azure": 0.65, "distributed systems": 0.80,
        }
    },
    "software_foundation": {
        "value": 0.50,
        "skills": {
            "python": 0.90, "fastapi": 0.75, "django": 0.70,
            "postgresql": 0.65, "mongodb": 0.65, "redis": 0.65,
            "rest api": 0.65, "system design": 0.70,
            "data structures": 0.60, "algorithms": 0.60,
        }
    },
    "wrong_domain": {
        "value": -0.30,  # Penalty cluster
        "skills": {
            "computer vision": -0.20, "image classification": -0.15,
            "object detection": -0.15, "yolo": -0.20, "opencv": -0.20,
            "speech recognition": -0.20, "asr": -0.20, "tts": -0.20,
            "slam": -0.30, "lidar": -0.30, "robotics": -0.30,
            "autonomous driving": -0.30,
            "photoshop": -0.10, "figma": -0.10, "illustrator": -0.10,
        }
    }
}


def _match_skill_to_cluster(skill_name: str) -> tuple:
    """Find which cluster a skill belongs to and its value within that cluster."""
    name_lower = skill_name.lower()
    best_cluster = None
    best_node_value = 0.0
    best_cluster_value = 0.0

    for cluster_name, cluster_data in SKILL_CLUSTERS.items():
        for skill_key, node_value in cluster_data["skills"].items():
            if skill_key in name_lower or name_lower in skill_key:
                abs_val = abs(node_value)
                if abs_val > abs(best_node_value):
                    best_cluster = cluster_name
                    best_node_value = node_value
                    best_cluster_value = cluster_data["value"]
                break

    return best_cluster, best_node_value, best_cluster_value


def _credibility_factor(skill: dict, assessment_scores: dict) -> float:
    """Compute how credible this skill claim is."""
    proficiency = skill.get("proficiency", "beginner")
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)

    prof_base = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85, "expert": 1.0}
    base = prof_base.get(proficiency, 0.4)

    # Endorsement validation (log scale)
    endorse_factor = min(math.log1p(endorsements) / math.log1p(30), 1.0)

    # Duration validation (36 months = fully proven)
    duration_factor = min(duration / 36, 1.0)

    # Assessment validation
    assessed = assessment_scores.get(skill.get("name", ""))
    if assessed is not None:
        expected_floor = {"beginner": 0, "intermediate": 30, "advanced": 55, "expert": 75}
        if assessed < expected_floor.get(proficiency, 0) - 15:
            assess_factor = assessed / 100  # Trust the assessment over claimed level
        else:
            assess_factor = max(0.5, assessed / 100)
    else:
        assess_factor = 0.75  # benefit of doubt

    credibility = (0.35 * base + 0.30 * assess_factor +
                   0.20 * endorse_factor + 0.15 * duration_factor)
    return credibility


def skill_graph_score(candidate: dict) -> dict:
    """
    Compute how deep the candidate is in the JD skill cluster graph.

    Returns:
        score (0-1): overall graph depth score
        cluster_coverage: which clusters they activate
        cluster_depths: depth score per cluster
        neighborhood_density: how interconnected their skills are
        top_graph_skills: their strongest graph-matched skills
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    if not skills:
        return {
            "score": 0.0, "cluster_coverage": [],
            "cluster_depths": {}, "neighborhood_density": 0.0,
            "top_graph_skills": [],
        }

    cluster_scores: Dict[str, List[float]] = {}
    graph_skills = []

    for skill in skills:
        cluster, node_value, cluster_value = _match_skill_to_cluster(skill.get("name", ""))
        if cluster is None:
            continue

        credibility = _credibility_factor(skill, assessment_scores)
        effective_value = node_value * credibility

        if cluster not in cluster_scores:
            cluster_scores[cluster] = []
        cluster_scores[cluster].append(effective_value)

        if node_value > 0:
            graph_skills.append({
                "name": skill.get("name"),
                "cluster": cluster,
                "node_value": round(node_value, 2),
                "credibility": round(credibility, 2),
                "effective": round(effective_value, 3),
            })

    # ── Cluster depth per cluster ─────────────────────────────────────────────
    cluster_depths = {}
    for cluster_name, values in cluster_scores.items():
        pos_values = [v for v in values if v > 0]
        neg_values = [v for v in values if v < 0]
        if pos_values:
            # Use top-3 with diminishing returns
            sorted_vals = sorted(pos_values, reverse=True)[:3]
            weights = [1.0, 0.6, 0.3][:len(sorted_vals)]
            depth = sum(v * w for v, w in zip(sorted_vals, weights)) / sum(weights)
        else:
            depth = 0.0
        if neg_values:
            depth += sum(neg_values) * 0.3  # apply penalty
        cluster_depths[cluster_name] = max(-1.0, min(1.0, depth))

    # ── Neighborhood density (how many clusters are activated?) ─────────────
    positive_clusters = [c for c, d in cluster_depths.items()
                         if d > 0 and c != "wrong_domain"]
    n_clusters = len(positive_clusters)
    # Bonus for activating multiple relevant clusters
    neighborhood_density = min(n_clusters / 4, 1.0)  # 4+ clusters = full density

    # ── Overall graph score ──────────────────────────────────────────────────
    # Priority: retrieval core + ranking > ml > production > infra > software
    PRIORITY = {
        "retrieval_core": 0.30,
        "ranking_recommendation": 0.25,
        "ml_engineering": 0.20,
        "production_ml": 0.12,
        "infra_scale": 0.08,
        "software_foundation": 0.05,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for cluster_name, weight in PRIORITY.items():
        depth = cluster_depths.get(cluster_name, 0.0)
        weighted_sum += weight * max(0.0, depth)
        total_weight += weight

    base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Density bonus (interconnected skills > isolated skills)
    density_bonus = neighborhood_density * 0.10

    # Wrong domain penalty
    wrong_penalty = max(0.0, -cluster_depths.get("wrong_domain", 0.0)) * 0.15

    final_score = max(0.0, min(1.0, base_score + density_bonus - wrong_penalty))

    top_skills = sorted(graph_skills, key=lambda x: x["effective"], reverse=True)[:6]

    return {
        "score": round(final_score, 4),
        "cluster_coverage": positive_clusters,
        "cluster_depths": {k: round(v, 3) for k, v in cluster_depths.items()},
        "neighborhood_density": round(neighborhood_density, 3),
        "top_graph_skills": top_skills,
        "has_retrieval_core": cluster_depths.get("retrieval_core", 0) > 0.3,
        "has_ranking": cluster_depths.get("ranking_recommendation", 0) > 0.3,
    }
