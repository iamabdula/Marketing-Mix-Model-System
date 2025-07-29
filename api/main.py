from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Marketing Mix Model API",
    description="API for training and predicting sales using MMM.",
    version="1.0.0",
)

app.include_router(router)
