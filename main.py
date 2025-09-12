# main.py
from fastapi import FastAPI
from routes import schema_routes

app = FastAPI(
    title="AI Powered SQL Assessment Platform",
    description="Backend for schema upload, question generation, and query validation",
    version="0.1.0"
)

# Root route
@app.get("/")
def read_root():
    return {"message": "AI SQL Assessment Platform is running 🚀"}

# Register routers
app.include_router(schema_routes.router, prefix="/schema", tags=["Schema"])
