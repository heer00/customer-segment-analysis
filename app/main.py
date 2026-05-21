"""
app/main.py
-----------
FastAPI application factory - the heart of the backend.

WHY APPLICATION FACTORY PATTERN:
  Instead of one giant file with everything, main.py only:
  1. Creates the FastAPI app
  2. Registers routers (routes plug in like LEGO bricks)
  3. Defines startup/shutdown events
  4. Sets global middleware

  Adding a new feature = create a new router file and register it here.
  That's it. No touching existing code. This is the Open/Closed Principle.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.model import get_model
from app.schemas.customer import HealthResponse
from app.api.routes import predict, customers
from app.db import Base, engine


# --- Lifespan Event Handler ---------------------------------------------------
# Runs setup code at startup and cleanup at shutdown.
# This replaces deprecated @app.on_event("startup") in modern FastAPI.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- STARTUP --
    print("\n" + "=" * 50)
    print(f"  {settings.app_name}")
    print(f"  Version : {settings.app_version}")
    print(f"  Debug   : {settings.debug}")
    print("=" * 50)

    # Initialize DB tables
    print("[DB] Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("[DB] Database tables initialized successfully.")

    # Pre-load the ML model so first request isn't slow
    get_model()
    print("[READY] Server started. Visit http://127.0.0.1:8000/docs\n")

    yield  # <- Server runs while suspended here

    # -- SHUTDOWN --
    print("\n[SHUTDOWN] Server shutting down. Cleaning up...")


# --- Create FastAPI App -------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## AI-Powered Customer Intelligence Platform

Predict customer segments using KMeans clustering and get
business-friendly labels for actionable insights.

### Features
- ML-powered customer segmentation
- 5 distinct business-friendly segment labels
- Sub-10ms prediction latency (model pre-loaded)
- Input validation with clear error messages

### Segments
| Cluster | Label | Profile |
|---------|-------|---------|
| 0 | Premium Customers | High Income, High Spending |
| 1 | Careful Spenders | High Income, Low Spending |
| 2 | Budget Shoppers | Low Income, High Spending |
| 3 | Low Engagement | Low Income, Low Spending |
| 4 | Average Customers | Medium Income, Medium Spending |
    """,
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# --- CORS Middleware ----------------------------------------------------------
# Allows the HTML dashboard (Phase 5) to call this API from the browser.
# In production, replace "*" with your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register Routers ---------------------------------------------------------
# All prediction and customer routes are under /api/v1
app.include_router(predict.router, prefix="/api/v1")
app.include_router(customers.router, prefix="/api/v1")


# --- Root Endpoint ------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint - confirms the server is running."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
    }


# --- Health Check -------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Liveness probe - used by load balancers and monitoring tools
    (Kubernetes, AWS ELB, Datadog) to verify the server is alive.
    """
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=model_loaded,
    )
