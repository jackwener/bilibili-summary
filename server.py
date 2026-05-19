#!/usr/bin/env python3
"""
FastAPI 后端服务器
提供 REST API + SSE 实时进度推送
"""

import os
import asyncio
import json
import time
import re
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from bilibili_api import channel_series as bili_channel_series, user as bili_user, video as bili_video

from summarize import extract_bvid, get_uid_by_name, get_user_videos, get_favorite_videos, sanitize_filename

import routes.deps as deps
from routes.deps import (
    BUNDLE_DIR, DATA_DIR,
    init_credential, init_ai_client,
    send_progress, progress_generator,
    process_single_video, run_batch, save_user_meta,
)
from routes.favorites import router as favorites_router
from routes.asr import router as asr_router
from routes.settings import router as settings_router
from routes.auth import router as auth_router


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_credential()
    init_ai_client()
    yield


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Bilibili 视频总结器", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BUNDLE_DIR / "static")), name="static")

# Include route modules
app.include_router(favorites_router)
app.include_router(asr_router)
app.include_router(settings_router)
app.include_router(auth_router)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class SummarizeURLRequest(BaseModel):
    urls: list[str] = Field(default_factory=list, min_length=1, max_length=200)
    model: str = ""
    concurrency: int = Field(default=12, ge=1, le=20)


class SummarizeUserRequest(BaseModel):
    user: str  # UID or name
    count: int = Field(default=50, ge=1, le=200)
    model: str = ""
    concurrency: int = Field(default=12, ge=1, le=20)


class SummarizeFavRequest(BaseModel):
    count: int = Field(default=20, ge=1, le=200)
    model: str = ""
    concurrency: int = Field(default=12, ge=1, le=20)


def _resolve_summary_file(path: str) -> Path | None:
    """Resolve a summary file path safely under DATA_DIR/summary."""
    summary_root = (DATA_DIR / "summary").resolve()
    try:
        target = (summary_root / path).resolve(strict=False)
    except (RuntimeError, ValueError):
        return None

    if summary_root not in target.parents:
        return None
    if not target.is_file():
        return None
    if target.suffix.lower() != ".md":
        return None
    return target


_BVID_RE = re.compile(r"\*\*BV号\*\*:\s*(BV[0-9A-Za-z]+)")
_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_cover_cache: dict[str, str] = {}
_MAX_COVER_LOOKUPS_PER_REQUEST = 40


def _dedupe_bvids(bvids: list[str]) -> list[str]:
    seen = set()
    result = []
    for bvid in bvids:
        if bvid and bvid not in seen:
            seen.add(bvid)
            result.append(bvid)
    return result


def _page_url(bvid: str, page_index: int) -> str:
    return f"https://www.bilibili.com/video/{bvid}?p={page_index + 1}"


def _extract_bvids_from_payload(payload) -> list[str]:
    bvids = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() == "bvid" and isinstance(value, str) and value.startswith("BV"):
                bvids.append(value)
            else:
                bvids.extend(_extract_bvids_from_payload(value))
    elif isinstance(payload, list):
        for item in payload:
            bvids.extend(_extract_bvids_from_payload(item))
    return _dedupe_bvids(bvids)


def _int_query_param(query: dict[str, list[str]], *names: str) -> int | None:
    for name in names:
        values = query.get(name)
        if values:
            try:
                return int(values[0])
            except (TypeError, ValueError):
                return None
    return None


async def _fetch_channel_bvids(channel_id: int, type_: bili_channel_series.ChannelSeriesType) -> list[str]:
    channel = bili_channel_series.ChannelSeries(type_=type_, id_=channel_id, credential=deps.credential)
    bvids = []

    for page in range(1, 51):
        data = await channel.get_videos(pn=page, ps=100)
        page_bvids = _extract_bvids_from_payload(data)
        before = len(bvids)
        bvids = _dedupe_bvids(bvids + page_bvids)
        if not page_bvids or len(bvids) == before:
            break
        if len(page_bvids) < 100:
            break

    return bvids


async def _expand_bvid_url(raw_url: str, bvid: str) -> list[str]:
    try:
        v = bili_video.Video(bvid=bvid, credential=deps.credential)
        info = await v.get_info()
    except Exception:
        return [bvid]

    collection_bvids = _extract_bvids_from_payload(info.get("ugc_season"))
    if collection_bvids:
        return collection_bvids

    try:
        pages = await v.get_pages()
    except Exception:
        return [bvid]

    if len(pages) > 1:
        return [_page_url(bvid, index) for index in range(len(pages))]
    return [bvid]


async def _resolve_input_bvids(raw_url: str) -> list[str]:
    raw_url = raw_url.strip()
    if not raw_url:
        return []

    parsed = urlparse(raw_url)
    query = parse_qs(parsed.query)
    path = parsed.path.lower()

    season_id = _int_query_param(query, "season_id", "seasonid")
    series_id = _int_query_param(query, "series_id", "seriesid")
    sid = _int_query_param(query, "sid")

    if season_id is not None:
        return await _fetch_channel_bvids(season_id, bili_channel_series.ChannelSeriesType.SEASON)
    if series_id is not None:
        return await _fetch_channel_bvids(series_id, bili_channel_series.ChannelSeriesType.SERIES)
    if sid is not None and "collectiondetail" in path:
        return await _fetch_channel_bvids(sid, bili_channel_series.ChannelSeriesType.SEASON)
    if sid is not None and "seriesdetail" in path:
        return await _fetch_channel_bvids(sid, bili_channel_series.ChannelSeriesType.SERIES)

    bvid = extract_bvid(raw_url)
    return await _expand_bvid_url(raw_url, bvid)


async def _resolve_input_bvids_many(urls: list[str]) -> list[str]:
    bvids = []
    for url in urls:
        bvids.extend(await _resolve_input_bvids(url))
    return _dedupe_bvids(bvids)


def _extract_summary_info(md_path: Path) -> tuple[str, str]:
    """Extract (bvid, title) from summary markdown content."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "", ""

    bvid_match = _BVID_RE.search(text)
    title_match = _TITLE_RE.search(text)
    bvid = bvid_match.group(1) if bvid_match else ""
    title = title_match.group(1).strip() if title_match else ""
    return bvid, title


def _build_summary_item(md_path: Path, summary_root: Path) -> dict:
    rel = md_path.relative_to(summary_root)
    item = {
        "name": md_path.stem,
        "path": str(rel),
        "no_subtitle": "no_subtitle" in str(rel),
        "bvid": "",
        "cover": "",
        "duration": 0,
        "author_name": "",
    }

    meta_path = md_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            item["name"] = meta.get("title", "") or item["name"]
            item["bvid"] = meta.get("bvid", "") or ""
            item["cover"] = meta.get("cover_url", "") or ""
            item["duration"] = meta.get("duration", 0) or 0
            item["author_name"] = meta.get("author_name", "") or ""
            if isinstance(item["cover"], str) and item["cover"].startswith("//"):
                item["cover"] = "https:" + item["cover"]
        except Exception:
            pass

    if not item["bvid"]:
        md_bvid, md_title = _extract_summary_info(md_path)
        item["bvid"] = md_bvid
        if md_title:
            item["name"] = md_title

    if item["bvid"] and item["bvid"] in _cover_cache and not item["cover"]:
        item["cover"] = _cover_cache[item["bvid"]]

    return item


async def _fetch_cover_by_bvid(bvid: str) -> str:
    if not bvid:
        return ""
    if bvid in _cover_cache:
        return _cover_cache[bvid]

    try:
        v = bili_video.Video(bvid=bvid, credential=deps.credential)
        info = await v.get_info()
        cover = info.get("pic", "") or ""
        if isinstance(cover, str) and cover.startswith("//"):
            cover = "https:" + cover
        _cover_cache[bvid] = cover
        return cover
    except Exception:
        _cover_cache[bvid] = ""
        return ""


async def _fill_missing_covers(items: list[dict]):
    candidates: list[str] = []
    seen = set()
    for item in items:
        bvid = item.get("bvid", "")
        if bvid and not item.get("cover") and bvid not in seen:
            seen.add(bvid)
            candidates.append(bvid)

    if not candidates:
        return

    sem = asyncio.Semaphore(6)
    targets = candidates[:_MAX_COVER_LOOKUPS_PER_REQUEST]

    async def bounded_fetch(bvid: str):
        async with sem:
            await _fetch_cover_by_bvid(bvid)

    await asyncio.gather(*[bounded_fetch(bv) for bv in targets])

    for item in items:
        bvid = item.get("bvid", "")
        if bvid and not item.get("cover"):
            item["cover"] = _cover_cache.get(bvid, "")


# ---------------------------------------------------------------------------
# Core API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return (BUNDLE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    return {"logged_in": deps.credential is not None, "ai_configured": deps.ai_client is not None}


@app.get("/api/summaries")
async def list_summaries():
    """List all generated summaries, structured by category."""
    summary_root = DATA_DIR / "summary"
    if not summary_root.exists():
        return {"categories": []}

    categories = []
    all_items: list[dict] = []

    # 1) Standalone
    standalone_dir = summary_root / "standalone"
    if standalone_dir.exists():
        items = []
        for md in sorted(standalone_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            item = _build_summary_item(md, summary_root)
            items.append(item)
            all_items.append(item)
        if items:
            categories.append({"type": "standalone", "label": "独立视频", "icon": "link", "count": len(items), "items": items})

    # 2) Favorites
    fav_dir = summary_root / "favorites"
    if fav_dir.exists():
        items = []
        for md in sorted(fav_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            item = _build_summary_item(md, summary_root)
            items.append(item)
            all_items.append(item)
        if items:
            categories.append({"type": "favorites", "label": "收藏", "icon": "star", "count": len(items), "items": items})

    # 3) Users — each UID is a sub-group with display name
    users_dir = summary_root / "users"
    if users_dir.exists():
        user_groups = []
        for uid_folder in sorted(users_dir.iterdir()):
            if not uid_folder.is_dir():
                continue
            uid = uid_folder.name
            meta_file = uid_folder / ".meta.json"
            display_name = uid
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    display_name = meta.get("name", uid)
                except Exception:
                    pass

            items = []
            for md in sorted(uid_folder.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
                item = _build_summary_item(md, summary_root)
                items.append(item)
                all_items.append(item)
            if items:
                user_groups.append({"uid": uid, "display_name": display_name, "count": len(items), "items": items})

        if user_groups:
            total = sum(g["count"] for g in user_groups)
            categories.append({"type": "users", "label": "UP 主", "icon": "users", "count": total, "groups": user_groups})

    await _fill_missing_covers(all_items)
    return {"categories": categories}


@app.get("/api/summary/{path:path}")
async def read_summary(path: str):
    filepath = _resolve_summary_file(path)
    if not filepath:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return {"content": filepath.read_text(encoding="utf-8"), "path": path}


@app.post("/api/summarize/url")
async def summarize_urls(req: SummarizeURLRequest):
    task_id = f"url-{int(time.time()*1000)}"
    try:
        bvids = await _resolve_input_bvids_many(req.urls)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"解析链接失败: {e}"})

    if not bvids:
        return JSONResponse(status_code=400, content={"error": "无法解析任何 BV 号"})

    model = req.model or deps.DEFAULT_MODEL
    asyncio.create_task(run_batch(bvids, model, req.concurrency, "standalone", task_id))
    return {"task_id": task_id, "total": len(bvids)}


@app.post("/api/summarize/user")
async def summarize_user(req: SummarizeUserRequest):
    task_id = f"user-{int(time.time()*1000)}"

    async def _run():
        username = None
        if req.user.isdigit():
            uid = int(req.user)
        else:
            username = req.user
            uid = await get_uid_by_name(req.user)
            if not uid:
                await send_progress(task_id, "error", {"message": f"未找到 UP 主: {req.user}"})
                await send_progress(task_id, "done", {"total": 0, "success": 0, "skipped": 0, "no_subtitle": 0, "errors": 1})
                return

        # Fetch user info and save metadata
        try:
            u = bili_user.User(uid=uid, credential=deps.credential)
            user_info = await u.get_user_info()
            resolved_name = user_info.get('name', username or str(uid))
        except Exception:
            resolved_name = username or str(uid)

        save_user_meta(uid, resolved_name)

        model = req.model or deps.DEFAULT_MODEL
        await send_progress(task_id, "info", {"message": f"获取 UP 主 {resolved_name} (UID:{uid}) 的最新 {req.count} 个视频..."})
        bvids = await get_user_videos(uid, req.count, deps.credential)

        if not bvids:
            await send_progress(task_id, "error", {"message": "未找到视频"})
            await send_progress(task_id, "done", {"total": 0, "success": 0, "skipped": 0, "no_subtitle": 0, "errors": 0})
            return

        await run_batch(bvids, model, req.concurrency, f"users/{uid}", task_id)

    asyncio.create_task(_run())
    return {"task_id": task_id}


@app.post("/api/summarize/favorites")
async def summarize_favorites(req: SummarizeFavRequest):
    if not deps.credential:
        return JSONResponse(status_code=401, content={"error": "未登录 Bilibili"})

    task_id = f"fav-{int(time.time()*1000)}"

    async def _run():
        model = req.model or deps.DEFAULT_MODEL
        await send_progress(task_id, "info", {"message": f"获取默认收藏夹的最新 {req.count} 个视频..."})
        bvids = await get_favorite_videos(req.count, deps.credential)

        if not bvids:
            await send_progress(task_id, "error", {"message": "未找到视频"})
            await send_progress(task_id, "done", {"total": 0, "success": 0, "skipped": 0, "no_subtitle": 0, "errors": 0})
            return

        await run_batch(bvids, model, req.concurrency, "favorites", task_id)

    asyncio.create_task(_run())
    return {"task_id": task_id}


@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str, request: Request):
    last_id = int(request.headers.get("Last-Event-ID", "-1"))
    return StreamingResponse(
        progress_generator(task_id, last_id=last_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18520)
