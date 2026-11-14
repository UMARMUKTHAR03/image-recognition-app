"""
FastAPI application entrypoint for Image Recognition App.
Configures CORS, registers API routers, and exposes OpenAPI documentation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="Image Recognition App – Deep Learning Computer Vision Platform",
    description="Production Computer Vision & Image Classification API powered by PyTorch MobileNetV3-Small.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for GitHub Pages, local dev, and public clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
