"""Entry point to run the FastAPI application server."""

import uvicorn
from app.config import get_settings


def start():
    """Runs uvicorn server with settings from environment / config."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    start()
