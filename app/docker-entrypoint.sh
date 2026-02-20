#!/bin/sh
# Run Streamlit, FastAPI, or both depending on APP_MODE.

set -e

case "$APP_MODE" in
  api)
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
    ;;
  both)
    # Run API in background, Streamlit in foreground (keeps container alive, logs visible)
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    ;;
  streamlit|*)
    exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    ;;
esac
