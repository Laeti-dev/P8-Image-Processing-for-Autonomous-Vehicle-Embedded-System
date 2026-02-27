import os

import uvicorn


def main() -> None:
    """
    Entry point for running the FastAPI app with uvicorn.

    Azure App Service injects the listening port via the PORT environment
    variable. For local development, this falls back to 8000.
    """
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    uvicorn.run("app.backend.api:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
