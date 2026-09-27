"""Offline K-Means training, safe JSON model format, and matching serving features."""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def raw_matrix(profiles, topics):
    rows = []
    for profile in profiles:
        rows.append([profile[t]["performance_proxy"] if t in profile else np.nan for t in topics])
    return np.asarray(rows, dtype=float)


class ClusterModel:
    def __init__(self, artifact):
        if artifact.get("schema_version") != 1:
            raise ValueError("Unsupported cluster model version")
        self.artifact = artifact
        self.topics = artifact["topics"]
        self.imputation = np.asarray(artifact["imputation"], dtype=float)
        self.mean = np.asarray(artifact["scaler_mean"], dtype=float)
        self.scale = np.asarray(artifact["scaler_scale"], dtype=float)
        self.centres = np.asarray(artifact["centres"], dtype=float)
        self.needs = np.asarray(artifact["cluster_topic_needs"], dtype=float)
        n, k = len(self.topics), len(self.centres)
        if (n < 2 or len(set(self.topics)) != n or k < 2 or self.imputation.shape != (n,)
                or self.mean.shape != (2*n,) or self.scale.shape != (2*n,)
                or self.centres.shape != (k, 2*n) or self.needs.shape != (k, n)):
            raise ValueError("Invalid cluster model dimensions")
        for array in (self.imputation, self.mean, self.scale, self.centres, self.needs):
            if not np.isfinite(array).all():
                raise ValueError("Non-finite cluster model values")
        if (self.scale <= 0).any() or (self.needs < 0).any() or (self.needs > 1).any():
            raise ValueError("Invalid cluster scales or need scores")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, profile):
        raw = raw_matrix([profile], self.topics)
        observed = ~np.isnan(raw)
        if observed.sum() < 2:
            return None
        features = np.concatenate([np.where(observed, raw, self.imputation), observed.astype(float)], axis=1)
        z = (features - self.mean) / self.scale
        group = int(np.argmin(np.sum((self.centres - z) ** 2, axis=1)))
        return group, dict(zip(self.topics, self.needs[group].tolist()))


def train(profiles, topics, seed=42):
    topics = list(topics)
    profiles = [p for p in profiles if len(set(p) & set(topics)) >= 2]
    if len(profiles) < 20 or len(topics) < 2:
        raise ValueError("Need at least 20 training students with at least two assessed topics each")
    raw = raw_matrix(profiles, topics)
    observed = ~np.isnan(raw)
    counts = observed.sum(axis=0)
    imputation = np.divide(np.nansum(raw, axis=0), counts,
                           out=np.full(len(topics), .5), where=counts > 0)
    features = np.concatenate([np.where(observed, raw, imputation), observed.astype(float)], axis=1)
    scaler = StandardScaler().fit(features)
    z = scaler.transform(features)
    distinct = len(np.unique(z, axis=0))
    candidates = []
    best = None
    for k in range(2, min(5, distinct, len(profiles)//3) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
        sizes = np.bincount(model.labels_, minlength=k)
        if sizes.min() < 3:
            continue
        score = float(silhouette_score(z, model.labels_, sample_size=min(2000, len(z)), random_state=seed))
        candidates.append({"k": k, "silhouette": score, "cluster_sizes": sizes.tolist()})
        if best is None or score > best[0]:
            best = score, model
    if best is None:
        raise ValueError("No valid non-degenerate clustering with at least three students per cluster")
    score, model = best
    cluster_needs = []
    for group in range(model.n_clusters):
        part = raw[model.labels_ == group]
        n = np.isfinite(part).sum(axis=0)
        means = np.divide(np.nansum(part, axis=0), n, out=imputation.copy(), where=n >= 3)
        cluster_needs.append((1 - means).tolist())
    artifact = {"schema_version": 1, "topics": topics, "imputation": imputation.tolist(),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "centres": model.cluster_centers_.tolist(), "cluster_topic_needs": cluster_needs,
                "training_students": len(profiles), "seed": seed, "candidates": candidates,
                "selected_k": model.n_clusters, "silhouette": score,
                "note": "Exploratory grouping of observed responses; not validated learning styles or recommendation quality."}
    ClusterModel(artifact)
    return artifact
