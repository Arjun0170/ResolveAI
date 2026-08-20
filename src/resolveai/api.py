from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .service import SupportIntelligenceService


RequestText = Annotated[str, Field(min_length=1, max_length=1_000)]


class RouteRequest(BaseModel):
    text: RequestText


class AssistRequest(BaseModel):
    text: RequestText
    top_k: int = Field(default=3, ge=1, le=5)
    use_llm: bool = False


class FeedbackRequest(BaseModel):
    trace_id: UUID
    rating: Literal["helpful", "unhelpful"]
    correct_intent: str | None = Field(default=None, max_length=80)


class RouteCandidate(BaseModel):
    label: str
    score: float


class RouteResponse(BaseModel):
    label: str
    confidence: float
    abstained: bool
    raw_label: str
    candidates: list[RouteCandidate]


class ScoreComponents(BaseModel):
    lexical: float
    neural: float
    route: float


class EvidenceItem(BaseModel):
    doc_id: str
    intent: str
    title: str
    summary: str
    guidance: str
    score: float
    score_components: ScoreComponents


class GenerationInfo(BaseModel):
    provider: str
    fallback_used: bool


class AssistResponse(BaseModel):
    trace_id: UUID
    status: Literal["resolved", "needs_human_review"]
    route: RouteResponse
    answer: str
    citations: list[str]
    evidence: list[EvidenceItem]
    generation: GenerationInfo
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_version: str


class FeedbackResponse(BaseModel):
    accepted: bool


def create_app(
    service: SupportIntelligenceService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            app.state.service = SupportIntelligenceService.load()
        yield

    app = FastAPI(
        title="ResolveAI Support Intelligence API",
        version="0.1.0",
        description=(
            "Confidence-aware request routing and citation-checked hybrid retrieval."
        ),
        lifespan=lifespan,
    )
    if service is not None:
        app.state.service = service

    def current_service(request: Request) -> SupportIntelligenceService:
        return request.app.state.service

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(request: Request) -> dict:
        info = current_service(request).model_info()
        return {"status": "healthy", "model_version": info["model_version"]}

    @app.get("/v1/model-info", tags=["operations"])
    def model_info(request: Request) -> dict:
        return current_service(request).model_info()

    @app.post("/v1/route", response_model=RouteResponse, tags=["inference"])
    def route(payload: RouteRequest, request: Request) -> dict:
        return current_service(request).route(payload.text)

    @app.post("/v1/assist", response_model=AssistResponse, tags=["inference"])
    def assist(payload: AssistRequest, request: Request) -> dict:
        return current_service(request).assist(
            payload.text,
            top_k=payload.top_k,
            use_llm=payload.use_llm,
        )

    @app.post(
        "/v1/feedback",
        response_model=FeedbackResponse,
        status_code=202,
        tags=["feedback"],
    )
    def feedback(payload: FeedbackRequest, request: Request) -> dict:
        try:
            current_service(request).record_feedback(
                str(payload.trace_id),
                payload.rating,
                payload.correct_intent,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"accepted": True}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
    def metrics(request: Request) -> str:
        return current_service(request).metrics.openmetrics()

    return app


app = create_app()
