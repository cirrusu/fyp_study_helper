import csv
import hashlib
import json
from pathlib import Path

from .schemas import Question, Resource, Topic


class Catalog:
    """Validate the entire catalogue once; requests never fetch external URLs."""

    def __init__(self, directory: Path):
        def unique(items, key):
            result = {getattr(item, key): item for item in items}
            if len(result) != len(items):
                raise ValueError(f"Duplicate {key} in catalogue")
            return result

        self.topics = unique(
            [Topic.model_validate(x) for x in json.loads((directory / "topics.json").read_text())],
            "topic_id",
        )
        rows = []
        with (directory / "resources.csv").open(encoding="utf-8-sig", newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                try:
                    row["topics"] = [x.strip() for x in row["topics"].split("|") if x.strip()]
                    for key in ("difficulty", "duration_minutes", "url", "content"):
                        row[key] = row.get(key) or None
                    rows.append(Resource.model_validate(row))
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"Invalid resources.csv row {n}: {exc}") from exc
        if not rows or not self.topics:
            raise ValueError("Catalogue needs at least one resource and topic")
        self.resources = unique(rows, "resource_id")
        self.questions = unique(
            [Question.model_validate(x) for x in json.loads((directory / "questions.json").read_text())],
            "question_id",
        )
        for resource in rows:
            self.validate_topics(resource.topics)
        for question in self.questions.values():
            self.validate_topics([question.topic_id])
        # Question changes invalidate old diagnostic evidence; resource additions do not.
        self.assessment_fingerprint = hashlib.sha256(
            (directory / "topics.json").read_bytes() + (directory / "questions.json").read_bytes()
        ).hexdigest()

    def validate_topics(self, topics):
        unknown = sorted(set(topics) - self.topics.keys())
        if unknown:
            raise ValueError(f"Unknown topics: {', '.join(unknown)}")
