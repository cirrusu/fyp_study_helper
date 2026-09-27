import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .catalog import Catalog
from .clustering import ClusterModel
from .recommender import Recommender
from .schemas import Feedback, QuizSubmission, RecommendationRequest, StudentCreate
from .storage import Store, SubmissionConflict

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    database: Path = ROOT / "var/study_helper.sqlite3"
    cluster_model: Path | None = None

    @classmethod
    def from_env(cls):
        model = os.environ.get("STUDY_HELPER_CLUSTER_MODEL")
        return cls(Path(os.environ.get("STUDY_HELPER_DATA_DIR", ROOT / "data")),
                   Path(os.environ.get("STUDY_HELPER_DB", ROOT / "var/study_helper.sqlite3")),
                   Path(model) if model else None)


bearer = HTTPBearer(auto_error=False)


def current_student(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})
    student = request.app.state.store.authenticate(credentials.credentials)
    if student is None:
        raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})
    return student


def create_app(settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.catalog = Catalog(settings.data_dir)
        clusters = ClusterModel.load(settings.cluster_model) if settings.cluster_model else None
        app.state.engine = Recommender(app.state.catalog, clusters)
        app.state.store = Store(settings.database, app.state.catalog.assessment_fingerprint)
        yield

    app = FastAPI(title="Study Helper API", version="1.0.0", lifespan=lifespan,
                  description="Local research prototype: TF-IDF, diagnostic topic need, optional offline K-Means. No LLM key required.")

    def validate_topics(topics):
        try:
            app.state.catalog.validate_topics(topics)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health", tags=["Public"])
    def health():
        return {"status": "ok", "resources": len(app.state.catalog.resources),
                "questions": len(app.state.catalog.questions), "cluster_model_loaded": app.state.engine.clusters is not None}

    @app.get("/v1/topics", tags=["Public"])
    def topics():
        return list(app.state.catalog.topics.values())

    @app.get("/v1/resources", tags=["Public"])
    def resources(topic: str | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        if topic:
            validate_topics([topic])
        rows = [r for r in app.state.catalog.resources.values() if topic is None or topic in r.topics]
        return {"total": len(rows), "items": sorted(rows, key=lambda r: r.resource_id)[offset:offset+limit]}

    @app.get("/v1/resources/{resource_id}", tags=["Public"])
    def resource(resource_id: str):
        result = app.state.catalog.resources.get(resource_id)
        if result is None:
            raise HTTPException(404, "Resource not found")
        return result

    @app.get("/v1/quiz/questions", tags=["Quiz"])
    def questions(topic: str | None = None):
        if topic:
            validate_topics([topic])
        return [q.model_dump(exclude={"correct_option", "explanation"}) for q in app.state.catalog.questions.values()
                if topic is None or q.topic_id == topic]

    @app.post("/v1/students", status_code=201, tags=["Student"])
    def register(profile: StudentCreate):
        validate_topics(profile.selected_topics)
        return app.state.store.create_student(profile)

    @app.get("/v1/me", tags=["Student"])
    def me(student=Depends(current_student)):
        return {**student, "topic_scores": app.state.store.topic_scores(student["student_id"]),
                "note": "Smoothed latest-question correctness is a performance proxy, not calibrated mastery."}

    @app.put("/v1/me", tags=["Student"])
    def update(profile: StudentCreate, student=Depends(current_student)):
        validate_topics(profile.selected_topics)
        return app.state.store.update_student(student["student_id"], profile)

    @app.post("/v1/quiz/attempts", tags=["Quiz"])
    def submit(submission: QuizSubmission, student=Depends(current_student)):
        for answer in submission.answers:
            q = app.state.catalog.questions.get(answer.question_id)
            if q is None or answer.selected_option not in q.options:
                raise HTTPException(422, f"Unknown question or option: {answer.question_id}")
        try:
            return app.state.store.submit(student["student_id"], submission, app.state.catalog.questions)
        except SubmissionConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/v1/recommendations", tags=["Recommendations"])
    def recommend(body: RecommendationRequest, student=Depends(current_student)):
        if body.topics:
            validate_topics(body.topics)
        sid = student["student_id"]
        return app.state.engine.recommend(student, app.state.store.topic_scores(sid), app.state.store.feedback_for(sid), body)

    @app.post("/v1/feedback", tags=["Recommendations"])
    def feedback(body: Feedback, student=Depends(current_student)):
        if body.resource_id not in app.state.catalog.resources:
            raise HTTPException(404, "Resource not found")
        return app.state.store.record_feedback(student["student_id"], body)

    return app


app = create_app()
