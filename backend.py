from resupply_engine.api import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("resupply_engine.api:app", host="127.0.0.1", port=8000, reload=False)
