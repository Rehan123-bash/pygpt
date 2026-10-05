"""
server.py - Self-hosted demo page + completion API (PROJECT.md section 13).

Serves a hand-built HTML page at /  and  POST /complete  for programmatic use
(the same endpoint the VS Code extension stretch goal consumes). Reuses the
model + sampling logic from app/app.py, so the two demos can never disagree.

    python server.py            # http://127.0.0.1:8000
"""

import os
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
import app as pygpt                     # loads the model once; reuses complete_code()

api = FastAPI(title="PyGPT")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "static")


class CompleteRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 96
    temperature: float = 0.4
    top_k: int = 50
    top_p: float = 0.95
    mode: str = "doctest-verified (8)"


@api.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@api.get("/info")
def info():
    return {"model_card": pygpt.model_card}


@api.post("/complete")
def complete(req: CompleteRequest):
    t0 = time.time()
    full, stats = pygpt.complete_code(req.prompt, req.max_new_tokens, req.temperature,
                                      req.top_k, req.top_p, req.mode)
    return {"prompt": req.prompt, "completion": full[len(req.prompt):],
            "stats": stats, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=8000, log_level="warning")
