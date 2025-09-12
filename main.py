# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import schema_routes

app = FastAPI(
    title="AI Powered SQL Assessment Platform",
    description="Backend for schema upload, question generation, and query validation",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
def read_root():
    return {"message": "AI SQL Assessment Platform is running "}

# Register routers
app.include_router(schema_routes.router, prefix="/schema", tags=["Schema"])
