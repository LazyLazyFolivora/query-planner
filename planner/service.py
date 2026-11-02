"""HTTP service: GET /health, POST /v1/decide. Start with: uvicorn planner.service:app"""
import hmac
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import baseline, policy
from .models import DecideRequest, DecideResponse

app = FastAPI(title="query-planner", docs_url=None, redoc_url=None, openapi_url=None)


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message})


@app.middleware("http")
async def check_token(request: Request, call_next):
    """Bearer SERVICE_TOKEN on everything except /health. Checked before the body is read."""
    if request.url.path.startswith("/v1/"):
        expected = os.environ.get("SERVICE_TOKEN", "")
        if not expected:
            return error(503, "token_not_configured", "SERVICE_TOKEN is not set on the server")
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {expected}".encode()):
            return error(401, "unauthorized", "missing or wrong Bearer token")
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    problems = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
    return error(422, "invalid_request", "; ".join(problems))


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    codes = {404: "not_found", 405: "method_not_allowed"}
    return error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    return error(500, "internal_error", type(exc).__name__)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    return (baseline if req.policy == "baseline" else policy).decide(req)
