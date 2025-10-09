import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import teacher_routes, student_routes, validate_routes
from database import engine, Base
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,https://sql-playground-bay.vercel.app"
).split(",")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(title="AI SQL Assessment Platform", version="1.0")


# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        # Test DB connection (do NOT create tables directly in production)
        async with engine.begin() as conn:
            await conn.run_sync(lambda conn: logger.info("DB connected successfully"))
        logger.info("Application startup complete.")
    except Exception as e:
        logger.critical(f"Failed to connect to DB on startup: {e}")
        raise e


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down.")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(teacher_routes.router)
app.include_router(student_routes.router)
app.include_router(validate_routes.router)

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Backend running successfully"}
