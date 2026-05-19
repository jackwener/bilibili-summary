"""
Settings & Model Selection routes.
"""

import os
import asyncio
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import aiohttp
from dotenv import set_key

from routes.deps import DATA_DIR, init_ai_client

router = APIRouter(prefix="/api", tags=["settings"])

TOKEN_PLAN_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/anthropic"

MIMO_CHAT_MODELS = [
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "mimo-v2-pro",
    "mimo-v2-omni",
    "mimo-v2-flash",
]


def _clean_value(value: str | None) -> str:
    return (value or "").strip().strip("'\"")


def _normalize_anthropic_base_url(base_url: str, auth_token: str = "") -> str:
    base_url = base_url.strip().rstrip("/")
    auth_token = _clean_value(auth_token)
    if not base_url:
        return TOKEN_PLAN_DEFAULT_BASE_URL if auth_token.startswith("tp-") else ""

    parsed = urlparse(base_url)
    if auth_token.startswith("tp-") and parsed.netloc == "api.xiaomimimo.com":
        return TOKEN_PLAN_DEFAULT_BASE_URL
    if parsed.netloc.startswith("token-plan-") and parsed.netloc.endswith(".xiaomimimo.com"):
        scheme = parsed.scheme or "https"
        return f"{scheme}://{parsed.netloc}/anthropic"
    if parsed.netloc == "api.xiaomimimo.com":
        scheme = parsed.scheme or "https"
        return f"{scheme}://{parsed.netloc}/anthropic"

    return base_url


def _known_models_for_base_url(base_url: str) -> Optional[list[dict]]:
    parsed = urlparse(base_url.strip())
    if parsed.netloc == "api.xiaomimimo.com" or (
        parsed.netloc.startswith("token-plan-") and parsed.netloc.endswith(".xiaomimimo.com")
    ):
        return [{"id": model, "owned_by": "xiaomi-mimo"} for model in MIMO_CHAT_MODELS]
    return None


@router.get("/settings")
async def get_settings():
    """Return current API settings (token partially masked)."""
    from routes.deps import DEFAULT_MODEL
    token = _clean_value(os.getenv('ANTHROPIC_AUTH_TOKEN') or os.getenv('MIMO_API_KEY'))
    masked = token[:8] + '***' + token[-4:] if len(token) > 12 else '***'
    return {
        "base_url": _clean_value(os.getenv('ANTHROPIC_BASE_URL')),
        "auth_token_masked": masked,
        "default_model": DEFAULT_MODEL,
    }


class SaveSettingsRequest(BaseModel):
    base_url: str = ""
    auth_token: str = ""  # empty = don't change
    default_model: str = ""


@router.post("/settings")
async def save_settings(req: SaveSettingsRequest):
    """Save API settings to .env.local and hot-reload ai_client."""
    import routes.deps as deps

    env_path = str(DATA_DIR / '.env.local')
    changed = []

    existing_token = _clean_value(os.getenv('ANTHROPIC_AUTH_TOKEN') or os.getenv('MIMO_API_KEY'))
    incoming_token = _clean_value(req.auth_token) if req.auth_token and '***' not in req.auth_token else existing_token
    base_url = _normalize_anthropic_base_url(req.base_url, incoming_token)
    if base_url:
        set_key(env_path, 'ANTHROPIC_BASE_URL', base_url)
        os.environ['ANTHROPIC_BASE_URL'] = base_url
        changed.append('base_url')

    if req.auth_token and '***' not in req.auth_token:
        set_key(env_path, 'ANTHROPIC_AUTH_TOKEN', incoming_token)
        os.environ['ANTHROPIC_AUTH_TOKEN'] = incoming_token
        changed.append('auth_token')

    if req.default_model:
        set_key(env_path, 'DEFAULT_MODEL', req.default_model)
        os.environ['DEFAULT_MODEL'] = req.default_model
        deps.DEFAULT_MODEL = req.default_model
        changed.append('default_model')

    # Hot-reload AI client
    init_ai_client()

    return {"success": True, "changed": changed}


@router.get("/models")
async def list_models():
    """Fetch available models from the API provider's /v1/models endpoint."""
    from routes.deps import DEFAULT_MODEL
    base_url = _clean_value(os.getenv('ANTHROPIC_BASE_URL'))
    token = _clean_value(os.getenv('ANTHROPIC_AUTH_TOKEN') or os.getenv('MIMO_API_KEY'))

    known_models = _known_models_for_base_url(base_url)
    if known_models is not None:
        return {"models": known_models, "current": DEFAULT_MODEL}

    if not base_url or not token:
        return JSONResponse(status_code=400, content={"error": "API 未配置"})

    models_url = base_url.rstrip('/')
    if models_url.endswith('/v1'):
        models_url = models_url[:-3]
    models_url = models_url.rstrip('/') + '/v1/models'

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {token}"}
            async with session.get(models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return JSONResponse(status_code=resp.status, content={"error": f"API 返回 {resp.status}: {text[:200]}"})
                data = await resp.json()
                models = []
                for m in data.get('data', []):
                    models.append({
                        "id": m.get('id', ''),
                        "owned_by": m.get('owned_by', ''),
                    })
                models.sort(key=lambda x: x['id'])
                return {"models": models, "current": DEFAULT_MODEL}
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "请求超时"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
