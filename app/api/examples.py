"""Example text endpoints — list and retrieve pre-built example texts.

GET /api/examples      — list all available examples (summary)
GET /api/examples/{id} — retrieve a single example (full text)

Requirements implemented: 4.1, 4.2, 4.4.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import ExampleFull, ExampleSummary
from app.services.example_service import ExampleService

router = APIRouter(prefix="/api", tags=["examples"])

# Module-level instance — loaded at startup via lifespan.
example_service = ExampleService()


@router.get("/examples", response_model=list[ExampleSummary])
async def list_examples() -> list[ExampleSummary]:
    """Return summary information for every loaded example text."""
    return example_service.list_examples()


@router.get("/examples/{example_id}", response_model=ExampleFull)
async def get_example(example_id: str) -> ExampleFull:
    """Return the full text of a specific example.

    Raises 404 with a Hebrew message if the example id is not found.
    """
    result = example_service.get_example(example_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"דוגמה לא נמצאה: {example_id}",
        )
    return result
