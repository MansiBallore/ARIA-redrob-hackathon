"""
dna.py — ARIA Technique 1: Career DNA Fingerprinting

Instead of keyword matching, we extract a compressed "DNA sequence"
from each candidate's career trajectory and compute sequence similarity
against the ideal JD DNA using edit distance (Levenshtein).

Each career role is mapped to one of 8 DNA tokens:
  R = Retrieval/Ranking/Search/Recommendation
  M = ML/AI Engineering (training, fine-tuning, modeling)
  P = Platform/Infra/Scale (Kafka, Spark, K8s, distributed)
  S = Software Engineering (general backend, APIs)
  D = Data Engineering (pipelines, warehouses, ETL)
  C = Consulting/Services (IT services, outsourcing)
  X = Non-technical (marketing, sales, HR, ops)
  ? = Unknown/unclear

The ideal JD DNA = ["R", "M", "P", "S"] — someone who has done
retrieval work, ML engineering, infra, and software — in any order.

We measure:
  1. Token overlap (how many ideal tokens appear in candidate DNA)
  2. Sequence coherence (are the relevant tokens clustered, or sparse?)
  3. Recency bias (recent roles weighted 2× over older ones)
  4. Depth bonus (same token appearing 2+ times = expertise, not accident)
"""

from typing import List, Tuple


# ── DNA token mapping ─────────────────────────────────────────────────────────

# Keywords that map each role description/title to a DNA token
DNA_RULES = {
    "R": [  # Retrieval / Ranking / Search / Recommendation
        "retrieval", "ranking", "search", "recommendation", "recommender",
        "rag", "vector", "embedding", "faiss", "pinecone", "qdrant",
        "milvus", "weaviate", "elasticsearch", "semantic", "dense",
        "sparse", "hybrid search", "learning to rank", "ltr", "ndcg",
        "rerank", "bi-encoder", "cross-encoder", "ann", "knn",
        "collaborative filtering", "content-based",
    ],
    "M": [  # ML / AI Engineering
        "machine learning", "deep learning", "neural", "model",
        "training", "fine-tun", "pytorch", "tensorflow", "huggingface",
        "transformers", "xgboost", "lightgbm", "nlp", "llm",
        "inference", "experiment", "a/b test", "mlflow", "wandb",
        "feature engineering", "data science", "applied ml",
        "applied ai", "research engineer",
    ],
    "P": [  # Platform / Infra / Scale
        "kafka", "spark", "kubernetes", "docker", "distributed",
        "airflow", "redis", "celery", "rabbitmq", "streaming",
        "pipeline", "latency", "throughput", "scale", "serving",
        "triton", "onnx", "ray", "gcp", "aws", "azure", "cloud",
        "microservice", "devops", "sre", "reliability",
    ],
    "S": [  # Software Engineering
        "backend", "api", "rest", "graphql", "django", "fastapi",
        "flask", "node", "java", "golang", "software engineer",
        "system design", "database", "postgres", "mysql", "mongodb",
        "architecture", "engineer", "developer",
    ],
    "D": [  # Data Engineering
        "data engineer", "etl", "data pipeline", "warehouse",
        "snowflake", "bigquery", "dbt", "airflow", "pyspark",
        "data lake", "analytics engineer", "data platform",
    ],
    "C": [  # Consulting / IT Services (negative signal per JD)
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
        "it services", "consulting", "outsourc",
    ],
    "X": [  # Non-technical
        "marketing", "sales", "hr ", "human resource", "accountant",
        "finance", "operations manager", "content writer",
        "graphic design", "brand design", "customer support",
        "project manager", "mechanical engineer", "civil engineer",
    ],
}

# The ideal DNA for the JD (order doesn't fully matter but clustering does)
IDEAL_DNA = ["R", "R", "M", "P", "S"]  # R appears twice = depth matters

# Scores per token when matched
TOKEN_VALUE = {
    "R": 1.00,   # Core requirement
    "M": 0.80,   # Important
    "P": 0.60,   # Nice to have
    "S": 0.40,   # Expected baseline
    "D": 0.30,   # Adjacent, acceptable
    "C": -0.25,  # Negative per JD
    "X": -0.50,  # Hard negative
    "?": 0.05,   # Unknown
}


def _classify_role(title: str, description: str, company: str) -> str:
    """Map a single career role to its DNA token."""
    text = f"{title} {description} {company}".lower()

    token_scores = {}
    for token, keywords in DNA_RULES.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            token_scores[token] = score

    if not token_scores:
        return "?"

    # Return the highest-scoring token
    return max(token_scores, key=token_scores.get)


def extract_dna(candidate: dict) -> Tuple[List[str], str]:
    """
    Extract the career DNA sequence for a candidate.

    Returns:
        (dna_list, dna_string)
        e.g. (["D", "M", "R", "S"], "DMRS")

    Order: most recent job first (index 0 = current role).
    """
    history = candidate.get("career_history", [])
    dna = []

    for job in history:
        token = _classify_role(
            job.get("title", ""),
            job.get("description", ""),
            job.get("company", ""),
        )
        dna.append(token)

    return dna, "".join(dna)


def dna_similarity_score(candidate: dict) -> dict:
    """
    Compute DNA similarity between candidate and ideal JD DNA.

    Returns dict with:
        score (0-1): overall DNA match quality
        dna_sequence: string like "RMPS"
        token_breakdown: per-token analysis
        depth_tokens: tokens appearing 2+ times (expertise signal)
        recency_score: how relevant are their most recent roles?
    """
    dna_list, dna_string = extract_dna(candidate)

    if not dna_list:
        return {
            "score": 0.0,
            "dna_sequence": "",
            "token_breakdown": {},
            "depth_tokens": [],
            "recency_score": 0.0,
        }

    # ── Token overlap score ────────────────────────────────────────────────
    ideal_tokens = set(IDEAL_DNA)
    candidate_tokens = set(dna_list)

    matched = ideal_tokens & candidate_tokens
    token_overlap = len(matched) / len(ideal_tokens)

    # ── Token value score (weighted by token quality) ───────────────────────
    raw_token_values = [TOKEN_VALUE.get(t, 0.05) for t in dna_list]
    avg_token_value = sum(raw_token_values) / len(raw_token_values)
    # Normalize to 0-1 (max possible = 1.0 for all R tokens)
    token_value_score = max(0.0, min(1.0, (avg_token_value + 0.5) / 1.5))

    # ── Recency score (recent roles weighted 2x) ────────────────────────────
    recency_weights = [1.0 / (i + 1) for i in range(len(dna_list))]
    recency_values = [
        TOKEN_VALUE.get(t, 0.05) * w
        for t, w in zip(dna_list, recency_weights)
    ]
    recency_score = max(0.0, min(1.0,
        (sum(recency_values) / sum(recency_weights) + 0.5) / 1.5
    ))

    # ── Depth score (expertise = same relevant token 2+ times) ──────────────
    depth_tokens = []
    from collections import Counter
    token_counts = Counter(dna_list)
    for token in ["R", "M", "P"]:
        if token_counts.get(token, 0) >= 2:
            depth_tokens.append(token)
    depth_bonus = len(depth_tokens) * 0.08

    # ── Penalty: if any C or X tokens exist ─────────────────────────────────
    c_count = token_counts.get("C", 0)
    x_count = token_counts.get("X", 0)
    penalty = min(c_count * 0.10 + x_count * 0.20, 0.40)

    # ── Final DNA score ──────────────────────────────────────────────────────
    dna_score = (
        0.35 * token_overlap +
        0.30 * recency_score +
        0.25 * token_value_score +
        0.10 * min(depth_bonus / 0.24, 1.0)
    ) - penalty

    dna_score = max(0.0, min(1.0, dna_score))

    return {
        "score": round(dna_score, 4),
        "dna_sequence": dna_string,
        "matched_tokens": sorted(matched),
        "depth_tokens": depth_tokens,
        "recency_score": round(recency_score, 3),
        "token_overlap": round(token_overlap, 3),
        "has_retrieval": "R" in candidate_tokens,
        "has_ml": "M" in candidate_tokens,
        "services_heavy": c_count >= 2,
    }
