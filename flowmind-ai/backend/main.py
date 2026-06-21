from dotenv import load_dotenv
load_dotenv()
 
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import events, predictions, resources, analytics, diversion, realtime, assistant, livedata
from ml.engine import get_df, get_models
import uvicorn

logger = logging.getLogger("flowmind.startup")
 
app = FastAPI(
    title="FlowMind AI — Traffic Command API",
    description="Intelligent Event Traffic Command Center for Bengaluru",
    version="2.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(resources.router, prefix="/api/resources", tags=["Resources"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(diversion.router, prefix="/api/diversion", tags=["Diversion"])
app.include_router(realtime.router, prefix="/api/realtime", tags=["Realtime"])
app.include_router(assistant.router, prefix="/api/assistant", tags=["Assistant"])
app.include_router(livedata.router, prefix="/api/live", tags=["Live Data"])

@app.on_event("startup")
async def warm_up_ml_engine():
    """
    Pre-load the dataset and pre-train the ML ensemble (GBR + RandomForest +
    GBR-delay + GBC) once, here, at server boot.

    Without this, get_models() trains lazily on first use — and the ONLY
    code path that touches get_models() is the /api/live/* fallback logic
    (sample_live_incidents / predict_corridor_congestion_now), which the
    Historical pages never call. That meant the *first* hit to the Live Map
    after a (re)start paid a multi-second, synchronous, CPU-bound training
    cost inline inside the request — blocking the asyncio event loop and
    frequently exceeding the frontend's request timeout, so live incidents/
    traffic came back empty while Historical (which only needs get_df())
    loaded fine. Training once here, before the app accepts traffic, removes
    that lazy/blocking path entirely.
    """
    try:
        get_df()
        get_models()
        logger.info("FlowMind ML engine warmed up — live endpoints ready.")
    except Exception as e:
        logger.exception("ML engine warm-up failed: %s", e)

@app.get("/")
async def root():
    return {"message": "FlowMind AI API v2.0 — Traffic Command Center", "status": "operational"}
 
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "FlowMind AI"}
 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)