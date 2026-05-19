"""
Shared state and dependencies for all route modules.
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv, set_key
import anthropic
from bilibili_api import video
from bilibili_api.utils.network import Credential

from summarize import (
    extract_bvid, get_subtitle, save_ass, save_summary,
    summarize_with_claude, sanitize_filename
)


# ---------------------------------------------------------------------------
# Path resolution (supports PyInstaller bundle)
# ---------------------------------------------------------------------------
BUNDLE_DIR = Path(os.environ.get('BILISUMMARY_BUNDLE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = Path(os.environ.get('BILISUMMARY_DATA_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv(str(DATA_DIR / '.env.local'))


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
credential: Optional[Credential] = None
ai_client: Optional[anthropic.AsyncAnthropic] = None
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mimo-v2.5-pro")


def clean_env_value(value: Optional[str]) -> str:
    return (value or "").strip().strip("'\"")


def init_credential():
    global credential
    sessdata = clean_env_value(os.getenv('BILIBILI_SESSION_TOKEN'))
    bili_jct = clean_env_value(os.getenv('BILIBILI_BILI_JCT'))
    ac_time_value = clean_env_value(os.getenv('BILIBILI_AC_TIME_VALUE'))
    if sessdata and bili_jct:
        credential = Credential(sessdata=sessdata, bili_jct=bili_jct, ac_time_value=ac_time_value or "")
        return True
    return False


def init_ai_client():
    global ai_client
    base_url = clean_env_value(os.getenv('ANTHROPIC_BASE_URL'))
    auth_token = clean_env_value(os.getenv('ANTHROPIC_AUTH_TOKEN')) or clean_env_value(os.getenv('MIMO_API_KEY'))
    if 'xiaomimimo.com' in base_url:
        ai_client = anthropic.AsyncAnthropic(base_url=base_url, auth_token=auth_token)
    else:
        ai_client = anthropic.AsyncAnthropic(base_url=base_url, api_key=auth_token)


# ---------------------------------------------------------------------------
# SSE Progress (event-history based, supports reconnection)
# ---------------------------------------------------------------------------
progress_tasks: dict[str, dict] = {}


def _ensure_task(task_id: str):
    if task_id not in progress_tasks:
        progress_tasks[task_id] = {
            "events": [],
            "notify": asyncio.Event(),
            "done": False,
        }


async def send_progress(task_id: str, event: str, data: dict):
    _ensure_task(task_id)
    task = progress_tasks[task_id]
    task["events"].append({"event": event, "data": data})
    if event == "done":
        task["done"] = True
        asyncio.get_event_loop().call_later(300, lambda: progress_tasks.pop(task_id, None))
    task["notify"].set()


async def progress_generator(task_id: str, last_id: int = -1):
    _ensure_task(task_id)
    cursor = last_id + 1

    while True:
        task = progress_tasks.get(task_id)
        if not task:
            break

        while cursor < len(task["events"]):
            msg = task["events"][cursor]
            yield f"id: {cursor}\nevent: {msg['event']}\ndata: {json.dumps(msg['data'], ensure_ascii=False)}\n\n"
            if msg["event"] == "done":
                return
            cursor += 1

        if task["done"]:
            break

        task["notify"].clear()
        try:
            await asyncio.wait_for(task["notify"].wait(), timeout=15)
        except asyncio.TimeoutError:
            yield ": heartbeat\n\n"


# ---------------------------------------------------------------------------
# No-subtitle retry logic
# ---------------------------------------------------------------------------
MAX_NOSUB_RETRIES = 3


def _retries_file(output_subdir: str) -> Path:
    return DATA_DIR / "summary" / output_subdir / "no_subtitle" / ".retries.json"


def get_retry_count(output_subdir: str, safe_title: str) -> int:
    path = _retries_file(output_subdir)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return data.get(safe_title, 0)
    except Exception:
        return 0


def increment_retry_count(output_subdir: str, safe_title: str):
    path = _retries_file(output_subdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    data[safe_title] = data.get(safe_title, 0) + 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def clear_retry_count(output_subdir: str, safe_title: str):
    path = _retries_file(output_subdir)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
        data.pop(safe_title, None)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core processing (with progress callbacks)
# ---------------------------------------------------------------------------
def _page_index_from_url(url: str) -> int:
    try:
        p_values = parse_qs(urlparse(url).query).get("p")
        if not p_values:
            return 0
        return max(int(p_values[0]) - 1, 0)
    except (TypeError, ValueError):
        return 0


def _target_to_url(target: str) -> str:
    if target.startswith("http://") or target.startswith("https://"):
        return target
    return f"https://www.bilibili.com/video/{target}"


async def process_single_video(url: str, model: str, output_subdir: str, task_id: str):
    """Process one video and send progress events."""
    bvid = extract_bvid(url)
    if not bvid:
        await send_progress(task_id, "error", {"message": f"无法提取 BV 号: {url}"})
        return None

    try:
        page_index = _page_index_from_url(url)
        v = video.Video(bvid=bvid, credential=credential)
        info = await v.get_info()
        base_title = info.get("title", bvid)
        title = base_title
        duration = info.get("duration", 0)
        cover_url = info.get("pic", "")
        owner = info.get("owner", {})
        author_name = owner.get("name", "")
        author_uid = owner.get("mid", 0)
        pages = await v.get_pages()
        if page_index >= len(pages):
            message = f"分P序号超出范围: P{page_index + 1}/{len(pages)}"
            await send_progress(task_id, "error", {
                "title": base_title, "bvid": bvid, "message": message
            })
            return {"title": base_title, "status": "error", "message": message}
        if len(pages) > 1:
            page = pages[page_index]
            part = (page.get("part") or "").strip()
            title = f"{base_title} - P{page_index + 1}"
            if part:
                title = f"{title} {part}"
            duration = page.get("duration") or duration
            url = f"https://www.bilibili.com/video/{bvid}?p={page_index + 1}"
        else:
            url = f"https://www.bilibili.com/video/{bvid}"

        safe_title = sanitize_filename(title)
        normal_path = DATA_DIR / "summary" / output_subdir / f"{safe_title}.md"
        nosub_path = DATA_DIR / "summary" / output_subdir / "no_subtitle" / f"{safe_title}.md"

        if normal_path.exists():
            await send_progress(task_id, "skip", {
                "title": title, "bvid": bvid,
                "path": f"{output_subdir}/{safe_title}.md"
            })
            return {"title": title, "status": "skipped"}

        if credential is None:
            message = "未登录 Bilibili，无法读取字幕；请先扫码登录后重试"
            await send_progress(task_id, "error", {
                "title": title, "bvid": bvid, "message": message
            })
            return {"title": title, "status": "error", "message": message}

        if nosub_path.exists():
            retries = get_retry_count(output_subdir, safe_title)
            if retries >= MAX_NOSUB_RETRIES:
                await send_progress(task_id, "skip", {
                    "title": title, "bvid": bvid,
                    "path": f"{output_subdir}/no_subtitle/{safe_title}.md"
                })
                return {"title": title, "status": "skipped"}
            else:
                await send_progress(task_id, "processing", {
                    "title": title, "bvid": bvid,
                    "step": f"重试获取字幕 ({retries+1}/{MAX_NOSUB_RETRIES})"
                })

        await send_progress(task_id, "processing", {"title": title, "bvid": bvid, "step": "获取字幕"})

        subtitle_text, subtitle_raw = await get_subtitle(v, page_index=page_index, pages=pages)

        if subtitle_raw:
            save_ass(title, subtitle_raw, output_subdir)

        await send_progress(task_id, "processing", {"title": title, "bvid": bvid, "step": "AI 生成总结"})

        summary, duration_sec = await summarize_with_claude(subtitle_text, title, ai_client, model=model)

        final_subdir = output_subdir
        if not subtitle_text:
            final_subdir = f"{output_subdir}/no_subtitle"
            increment_retry_count(output_subdir, safe_title)
        else:
            if nosub_path.exists():
                nosub_path.unlink()
                clear_retry_count(output_subdir, safe_title)

        save_summary(
            title, bvid, url, duration, summary, final_subdir,
            author_name=author_name, author_uid=author_uid, cover_url=cover_url
        )

        status = "no_subtitle" if not subtitle_text else "success"
        await send_progress(task_id, "completed", {
            "title": title, "bvid": bvid,
            "duration_sec": round(duration_sec, 2),
            "status": status,
            "path": f"{final_subdir}/{safe_title}.md"
        })
        return {"title": title, "status": status, "duration_sec": round(duration_sec, 2)}

    except Exception as e:
        await send_progress(task_id, "error", {"title": bvid, "message": str(e)})
        return {"title": bvid, "status": "error", "message": str(e)}


async def run_batch(bvids: list[str], model: str, concurrency: int, output_subdir: str, task_id: str):
    sem = asyncio.Semaphore(concurrency)
    results = []

    await send_progress(task_id, "start", {
        "total": len(bvids), "concurrency": concurrency, "model": model
    })

    async def bounded(bvid):
        async with sem:
            url = _target_to_url(bvid)
            try:
                r = await process_single_video(url, model, output_subdir, task_id)
                results.append(r)
            except Exception as e:
                await send_progress(task_id, "error", {"title": bvid, "message": str(e)})
                results.append({"title": bvid, "status": "error", "message": str(e)})

    try:
        await asyncio.gather(*[bounded(bv) for bv in bvids])
    except Exception as e:
        await send_progress(task_id, "error", {"title": "", "message": f"批处理异常: {e}"})

    success = sum(1 for r in results if r and r.get("status") == "success")
    skipped = sum(1 for r in results if r and r.get("status") == "skipped")
    no_sub = sum(1 for r in results if r and r.get("status") == "no_subtitle")
    errors = sum(1 for r in results if r and r.get("status") == "error")

    await send_progress(task_id, "done", {
        "total": len(bvids), "success": success, "skipped": skipped,
        "no_subtitle": no_sub, "errors": errors
    })
    return results


def save_user_meta(uid: int, name: str):
    """Save .meta.json in user summary directory for display name resolution."""
    user_dir = DATA_DIR / "summary" / "users" / str(uid)
    user_dir.mkdir(parents=True, exist_ok=True)
    meta_file = user_dir / ".meta.json"
    meta_file.write_text(json.dumps({"uid": uid, "name": name}, ensure_ascii=False), encoding="utf-8")
