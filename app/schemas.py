from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
Format = Literal["reading", "exercise", "video", "course"]
Level = Literal["beginner", "intermediate", "advanced"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Topic(Schema):
    topic_id: Identifier
    title: str = Field(min_length=1, max_length=150)
    description: str = Field(min_length=1, max_length=2000)


class Resource(Schema):
    resource_id: Identifier
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=20000)
    topics: list[Identifier] = Field(min_length=1, max_length=30)
    format: Format
    difficulty: Level | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=100000)
    duration_is_estimate: bool = False
    url: HttpUrl | None = None
    content: str | None = Field(default=None, max_length=20000)
    source: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def check_resource(self):
        if not self.url and not self.content:
            raise ValueError("A resource needs a URL or inline content")
        if len(set(self.topics)) != len(self.topics):
            raise ValueError("Duplicate resource topics")
        if self.duration_is_estimate and self.duration_minutes is None:
            raise ValueError("An estimated duration needs a value")
        return self


class Question(Schema):
    question_id: Identifier
    topic_id: Identifier
    prompt: str = Field(min_length=1, max_length=5000)
    options: dict[Identifier, str] = Field(min_length=2, max_length=6)
    correct_option: Identifier
    explanation: str

    @model_validator(mode="after")
    def valid_key(self):
        if self.correct_option not in self.options:
            raise ValueError("Correct option must exist in options")
        return self


class StudentCreate(Schema):
    selected_topics: list[Identifier] = Field(min_length=1, max_length=30)
    preferred_format: Format | None = None
    target_level: Level | None = None

    @model_validator(mode="after")
    def unique_topics(self):
        if len(set(self.selected_topics)) != len(self.selected_topics):
            raise ValueError("Duplicate selected topics")
        return self


class Answer(Schema):
    question_id: Identifier
    selected_option: Identifier


class QuizSubmission(Schema):
    submission_id: Identifier
    answers: list[Answer] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_questions(self):
        ids = [a.question_id for a in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("A question can appear only once per submission")
        return self


class Feedback(Schema):
    resource_id: Identifier
    helpful: bool | None = None
    completed: bool | None = None

    @model_validator(mode="after")
    def has_feedback(self):
        if self.helpful is None and self.completed is None:
            raise ValueError("Provide helpful and/or completed")
        return self


class RecommendationRequest(Schema):
    topics: list[Identifier] | None = Field(default=None, min_length=1, max_length=30)
    k: int = Field(default=5, ge=1, le=20)
    max_minutes: int | None = Field(default=None, ge=1, le=1440)
    include_completed: bool = False
    include_dismissed: bool = False
    algorithm: Literal["content", "performance", "cluster"] = "cluster"

    @model_validator(mode="after")
    def unique_topics(self):
        if self.topics and len(self.topics) != len(set(self.topics)):
            raise ValueError("Duplicate request topics")
        return self
