"""FastAPI application entry point.

Configures CORS, registers all API routers, and manages the
application lifespan (model and example loading at startup,
cleanup at shutdown).

Requirements implemented: 1.6, 15.5.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load services at startup and clean up at shutdown."""
    # Startup — load heavy resources into memory
    from app.api.analyze import model_service
    from app.api.examples import example_service

    model_service.load()
    example_service.load()
    yield
    # Shutdown — release resources
    model_service.unload()


app = FastAPI(title="Hebrew Writing Coach", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS — allow the configured frontend origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register API routers
# ---------------------------------------------------------------------------
from app.api.analyze import router as analyze_router
from app.api.revise import router as revise_router
from app.api.rewrite import router as rewrite_router
from app.api.examples import router as examples_router
from app.api.admin import router as admin_router
from app.api.health import router as health_router
from app.api.exercise import router as exercise_router

app.include_router(analyze_router)
app.include_router(revise_router)
app.include_router(rewrite_router)
app.include_router(exercise_router)
app.include_router(examples_router)
app.include_router(admin_router)
app.include_router(health_router)
