from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import math
from app.api.endpoints import router as api_router
from app.config import API_TITLE, API_VERSION, ALLOWED_ORIGINS, MAX_CONTENT_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware with strict allowed origins configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

from fastapi.encoders import jsonable_encoder

def sanitize_validation_error_input(obj):
    """
    Recursively replaces non-finite float values (NaN, Inf, -Inf) in validation error inputs
    with their string representations to prevent Starlette serialization errors.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj)
        return obj
    elif isinstance(obj, list):
        return [sanitize_validation_error_input(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_validation_error_input(v) for k, v in obj.items()}
    return obj

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for RequestValidationError to prevent Starlette from throwing a 500
    error when trying to serialize non-finite floats (NaN, Inf) back to the client.
    """
    # Use standard FastAPI encoder to serialize complex objects like Exception contexts first
    errors = jsonable_encoder(exc.errors())
    
    # Sanitize the input fields in validation errors
    for err in errors:
        if "input" in err:
            err["input"] = sanitize_validation_error_input(err["input"])
        
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors}
    )

# Middleware to limit maximum request size, preventing Denial of Service (DoS) payload attacks
@app.middleware("http")
async def limit_payload_size_middleware(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > MAX_CONTENT_LENGTH:
                    return Response(
                        content="Payload size exceeds permitted limit.",
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                    )
            except ValueError:
                return Response(
                    content="Malformed Content-Length header.",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
    return await call_next(request)

# Root route for standard landing check
@app.get("/")
def read_root():
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs"
    }

# Register API routes
app.include_router(api_router)
