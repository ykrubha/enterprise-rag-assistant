"""FastAPI service.

    KB_INDEX_DIR=data/index uvicorn kb_assistant.api:app --port 8000
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .assistant import AbstainPolicy, Assistant, Generator, LLMGenerator
from .index import HybridIndex, MiniLmEmbedder, MsMarcoReranker

logger = logging.getLogger("kb_assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    categories: list[str] | None = Field(
        default=None, description="Restrict retrieval to these collections, such as storage or security."
    )


class SearchRequest(AskRequest):
    k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="rerank", pattern="^(rerank|hybrid|bm25|dense)$")


class CitationOut(BaseModel):
    number: int
    title: str
    section: str
    source_url: str
    doc_id: str
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    citations: list[CitationOut]
    confidence: dict
    latency_ms: float


def load_policy(index_dir: Path) -> AbstainPolicy:
    """Use thresholds saved by the evaluation pipeline if there are any."""
    path = index_dir / "abstain_policy.json"
    if path.exists():
        return AbstainPolicy(**json.loads(path.read_text()))
    return AbstainPolicy()


def default_generator() -> Generator | None:
    provider = os.environ.get("KB_LLM_PROVIDER")
    if not provider:
        return None
    return LLMGenerator(provider=provider, model=os.environ.get("KB_LLM_MODEL"))


def create_app(assistant: Assistant | None = None, api_key: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if assistant is not None:
            app.state.assistant = assistant
        else:
            directory = Path(os.environ.get("KB_INDEX_DIR", "data/index"))
            index = HybridIndex.load(directory, MiniLmEmbedder(), MsMarcoReranker())
            app.state.assistant = Assistant(
                index, generator=default_generator(), policy=load_policy(directory)
            )
        yield

    app = FastAPI(title="Enterprise AI Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    required_key = api_key if api_key is not None else os.environ.get("KB_API_KEY")

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if required_key and not hmac.compare_digest(x_api_key or "", required_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    def known_categories(request: Request, categories: list[str] | None) -> list[str] | None:
        if categories is None:
            return None
        valid = {c.category for c in request.app.state.assistant.index.chunks}
        unknown = sorted(set(categories) - valid)
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown categories: {', '.join(unknown)}")
        return categories

    @app.get("/health")
    def health(request: Request) -> dict:
        index = request.app.state.assistant.index
        return {"status": "ok", "chunks": len(index.chunks),
                "documents": len({c.doc_id for c in index.chunks})}

    @app.get("/categories", dependencies=[Depends(check_key)])
    def categories(request: Request) -> dict:
        index = request.app.state.assistant.index
        return {"categories": sorted({c.category for c in index.chunks})}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(check_key)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        chosen = known_categories(request, body.categories)
        started = time.perf_counter()
        try:
            answer = request.app.state.assistant.ask(body.question, categories=chosen)
        except Exception as err:
            logger.exception("Answer generation failed")
            raise HTTPException(status_code=502, detail="The answer service failed.") from err
        logger.info("ask abstained=%s ms=%.1f", answer.abstained, (time.perf_counter() - started) * 1000)
        return AskResponse(**answer.to_dict())

    @app.post("/search", dependencies=[Depends(check_key)])
    def search(body: SearchRequest, request: Request) -> dict:
        chosen = known_categories(request, body.categories)
        index = request.app.state.assistant.index
        hits = index.search(body.question, k=body.k, mode=body.mode, categories=chosen)
        return {"results": [
            {"chunk_id": h.chunk.id, "title": h.chunk.title, "section": h.chunk.section,
             "source_url": h.chunk.source_url, "score": round(h.score, 5), "text": h.chunk.text}
            for h in hits
        ]}

    return app


app = create_app()
