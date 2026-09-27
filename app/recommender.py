import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LEVELS = {"beginner": 0, "intermediate": 1, "advanced": 2}


class Recommender:
    def __init__(self, catalog, clusters=None):
        self.catalog, self.clusters = catalog, clusters
        self.resources = sorted(catalog.resources.values(), key=lambda r: r.resource_id)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        texts = [" ".join([r.title, r.description, *[catalog.topics[t].description for t in r.topics]]) for r in self.resources]
        self.matrix = self.vectorizer.fit_transform(texts)
        if clusters and not set(clusters.topics).issubset(catalog.topics):
            raise ValueError("Cluster model topic IDs do not match this catalogue")

    def recommend(self, student, profile, feedback, request):
        topics = request.topics or student["selected_topics"]
        self.catalog.validate_topics(topics)
        # Fixed query across algorithms: diagnostic scores enter only through the need component.
        query = " ".join(self.catalog.topics[t].description for t in topics)
        vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(vector, self.matrix).ravel()
        group = self.clusters.predict(profile) if self.clusters and request.algorithm == "cluster" else None
        effective = request.algorithm
        notices = []
        if request.algorithm == "cluster" and group is None:
            effective = "performance"
            notices.append("Cluster component unavailable: use an offline model and assess at least two model topics.")
        if not profile:
            notices.append("No diagnostic evidence yet; unassessed topic need is neutral, not a diagnosed weakness.")
        results = []
        for idx, resource in enumerate(self.resources):
            matched_topics = sorted(set(topics) & set(resource.topics))
            if not matched_topics:
                continue
            history = feedback.get(resource.resource_id, {})
            if history.get("completed") and not request.include_completed:
                continue
            if history.get("helpful") == 0 and not request.include_dismissed:
                continue
            if request.max_minutes is not None and (resource.duration_minutes is None or resource.duration_minutes > request.max_minutes):
                continue
            # Score requested topics only: unrelated weak topics cannot boost a candidate.
            assessed = [profile[t] for t in matched_topics if t in profile]
            components = {"content": float(similarity[idx]),
                          "topic_need": float(np.mean([1 - x["performance_proxy"] for x in assessed])) if assessed else .5}
            weights = {"content": .35, "topic_need": .35}
            reasons = ["Matches requested topic(s): " + ", ".join(matched_topics) + "."]
            evidence = {t: profile[t] for t in matched_topics if t in profile}
            if effective == "content":
                weights = {"content": 1.0}
                components = {"content": components["content"]}
            else:
                if evidence:
                    for t, info in evidence.items():
                        reasons.append(f"{t}: latest answers to {info['attempted']} distinct question(s), {info['correct']} correct.")
                else:
                    reasons.append("No diagnostic evidence for these topics; topic need is neutral.")
                target = student["target_level"]
                if target and resource.difficulty:
                    components["level"] = max(0., 1 - .5*abs(LEVELS[target] - LEVELS[resource.difficulty]))
                    weights["level"] = .15
                    if resource.difficulty == target:
                        reasons.append("Matches your selected difficulty.")
                if student["preferred_format"]:
                    components["format"] = float(resource.format == student["preferred_format"])
                    weights["format"] = .05
                    if components["format"]:
                        reasons.append("Matches your preferred format.")
                if group:
                    needs = [group[1][t] for t in matched_topics if t in group[1]]
                    if needs:
                        components["cluster"] = float(np.mean(needs))
                        weights["cluster"] = .10
                        reasons.append("Includes aggregate topic need from an offline student cluster.")
            total = sum(weights.values())
            weights = {key: value/total for key, value in weights.items()}
            score = sum(components[key]*weight for key, weight in weights.items())
            if resource.duration_minutes is not None:
                reasons.append(f"Duration: {resource.duration_minutes} minutes" + (" (curator estimate)." if resource.duration_is_estimate else "."))
            results.append({"resource": resource.model_dump(mode="json"), "score": score,
                            "components": components, "weights": weights, "reasons": reasons,
                            "diagnostic_evidence": evidence})
        results.sort(key=lambda x: (-x["score"], x["resource"]["resource_id"]))
        if len(results) < request.k:
            notices.append(f"Only {len(results)} resources satisfy your filters; returning fewer than requested.")
        if request.max_minutes is not None:
            notices.append("Unknown durations are excluded. This is a per-resource limit, not a total-session budget.")
        return {"requested_algorithm": request.algorithm, "effective_algorithm": effective,
                "cluster_id": group[0] if group else None, "candidate_count": len(results),
                "recommendations": results[:request.k], "notices": notices,
                "score_note": "Ranking scores are heuristic relevance signals, not probabilities of learning success."}
