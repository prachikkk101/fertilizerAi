from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Fertilizer AI API",
    description="API for fertilizer recommendation and mixing parameters",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Fertilizer AI API is running"
    }