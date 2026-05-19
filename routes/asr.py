"""
ASR-based Summarization routes (for videos without subtitles).
"""

import os
import json
import asyncio
import tempfile

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

import aiohttp
from bilibili_api import video
from bilibili_api.video import VideoDownloadURLDataDetecter, AudioQuality

from routes.deps import (
    DATA_DIR,
    credential, ai_client, DEFAULT_MODEL,
    sanitize_filename,
)
from summarize import summarize_with_claude, save_summary

router = APIRouter(prefix="/api", tags=["asr"])


def _new_temp_path(suffix: str) -> str:
    """Create a unique temp path without race-prone mktemp()."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        return tmp.name


@router.post("/asr-summarize/{bvid}")
async def asr_summarize(bvid: str, output_subdir: str = ""):
    """Download audio → GLM-ASR transcription → LLM summary via SSE."""
    from routes.deps import credential, ai_client, DEFAULT_MODEL

    if not credential:
        return JSONResponse(status_code=401, content={"error": "未登录 Bilibili"})
    if not ai_client:
        return JSONResponse(status_code=400, content={"error": "未配置 AI API"})

    async def event_stream():
        try:
            # Step 1: Get video info
            yield f"data: {json.dumps({'step': 'info', 'message': '获取视频信息...'})}\n\n"
            v = video.Video(bvid=bvid, credential=credential)
            info = await v.get_info()
            title = info.get("title", bvid)
            duration = info.get("duration", 0)
            cover_url = info.get("pic", "")
            owner = info.get("owner", {})
            author_name = owner.get("name", "")
            author_uid = owner.get("mid", 0)
            safe_title = sanitize_filename(title)
            url = f"https://www.bilibili.com/video/{bvid}"

            # Auto-detect output_subdir if not provided
            nonlocal output_subdir
            if not output_subdir:
                summary_root = DATA_DIR / "summary"
                for subdir in ["standalone", "favorites"]:
                    if (summary_root / subdir / f"{safe_title}.md").exists() or \
                       (summary_root / subdir / "no_subtitle" / f"{safe_title}.md").exists():
                        output_subdir = subdir
                        break
                if not output_subdir:
                    users_dir = summary_root / "users"
                    if users_dir.exists():
                        for uid_folder in users_dir.iterdir():
                            if uid_folder.is_dir():
                                if (uid_folder / f"{safe_title}.md").exists() or \
                                   (uid_folder / "no_subtitle" / f"{safe_title}.md").exists():
                                    output_subdir = f"users/{uid_folder.name}"
                                    break
                if not output_subdir:
                    output_subdir = "standalone"

            # Step 2: Get audio download URL (use lowest quality to minimize size)
            yield f"data: {json.dumps({'step': 'audio_url', 'message': '获取音频流地址...'})}\n\n"
            download_data = await v.get_download_url(page_index=0)
            detector = VideoDownloadURLDataDetecter(download_data)
            streams = detector.detect_best_streams(
                audio_max_quality=AudioQuality._64K,
                no_dolby_audio=True,
                no_hires=True,
            )

            # detect_best_streams returns:
            # - DASH: [VideoStreamDownloadURL, AudioStreamDownloadURL | None]
            # - FLV/MP4: [FLVStreamDownloadURL] or [MP4StreamDownloadURL]
            audio_url = None
            if detector.check_flv_mp4_stream():
                # FLV/MP4: combined audio+video stream
                if streams and streams[0] and hasattr(streams[0], 'url'):
                    audio_url = streams[0].url
                    print(f"[ASR] Using FLV/MP4 stream")
            else:
                # DASH: audio is at index 1
                if len(streams) >= 2 and streams[1] is not None and hasattr(streams[1], 'url'):
                    audio_url = streams[1].url
                    print(f"[ASR] Using DASH audio stream")
                else:
                    for s in streams:
                        if s is not None and hasattr(s, 'audio_quality'):
                            audio_url = s.url
                            print(f"[ASR] Using fallback audio stream")
                            break

            if not audio_url:
                yield f"data: {json.dumps({'step': 'error', 'message': '无法获取音频流（可能是会员专属视频）'})}\n\n"
                return

            # Step 3: Download audio (with retry)
            yield f"data: {json.dumps({'step': 'download', 'message': '下载音频中...'})}\n\n"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": "https://www.bilibili.com",
            }
            audio_data = b""
            for dl_attempt in range(3):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(audio_url, headers=headers) as resp:
                            if resp.status == 200:
                                audio_data = await resp.read()
                                break
                            elif dl_attempt < 2:
                                yield f"data: {json.dumps({'step': 'download', 'message': f'HTTP {resp.status}，重试下载...'})}\n\n"
                                await asyncio.sleep(2)
                            else:
                                yield f"data: {json.dumps({'step': 'error', 'message': f'音频下载失败: HTTP {resp.status}'})}\n\n"
                                return
                except (aiohttp.ClientError, asyncio.TimeoutError) as dl_err:
                    if dl_attempt < 2:
                        yield f"data: {json.dumps({'step': 'download', 'message': '网络错误，重试下载...'})}\n\n"
                        await asyncio.sleep(2)
                    else:
                        yield f"data: {json.dumps({'step': 'error', 'message': f'音频下载失败: {dl_err}'})}\n\n"
                        return

            audio_size_mb = len(audio_data) / (1024 * 1024)
            yield f"data: {json.dumps({'step': 'downloaded', 'message': f'音频下载完成 ({audio_size_mb:.1f} MB)'})}\n\n"

            # Step 4: Send to GLM-ASR for transcription
            yield f"data: {json.dumps({'step': 'asr', 'message': '语音识别中 (GLM-ASR)...'})}\n\n"

            asr_endpoint = os.getenv(
                'GLM_ASR_BASE_URL',
                'https://open.bigmodel.cn/api/paas/v4/audio/transcriptions',
            )
            api_key = os.getenv('GLM_ASR_AUTH_TOKEN') or os.getenv('GLM_AUTH_TOKEN') or ''
            if not api_key and 'bigmodel.cn' in os.getenv('ANTHROPIC_BASE_URL', ''):
                api_key = os.getenv('ANTHROPIC_AUTH_TOKEN', '')
            if not api_key:
                yield f"data: {json.dumps({'step': 'error', 'message': '未配置 ASR Token。无字幕视频需要 GLM_ASR_AUTH_TOKEN；当前 Mimo 文本模型配置不能直接替代 GLM-ASR。'})}\n\n"
                return

            # Write audio to temp file for conversion
            m4s_path = _new_temp_path(".m4s")
            with open(m4s_path, 'wb') as f:
                f.write(audio_data)

            # Convert m4s to 29-second wav segments using PyAV (ASR limit: 30s per file)
            SEGMENT_DURATION = 29  # seconds, stay under 30s limit
            yield f"data: {json.dumps({'step': 'asr', 'message': '切分音频并转换格式...'})}\n\n"

            chunk_paths = []
            try:
                import av as pyav
                input_container = pyav.open(m4s_path)
                audio_stream_info = input_container.streams.audio[0]
                total_duration = float(audio_stream_info.duration * audio_stream_info.time_base) if audio_stream_info.duration else duration
                num_segments = max(1, int(total_duration / SEGMENT_DURATION) + (1 if total_duration % SEGMENT_DURATION > 0.5 else 0))

                print(f"[ASR] Audio duration: {total_duration:.1f}s, splitting into ~{num_segments} segments of {SEGMENT_DURATION}s")

                # Decode all audio frames first
                all_frames = []
                for frame in input_container.decode(audio=0):
                    all_frames.append(frame)
                input_container.close()

                if not all_frames:
                    yield f"data: {json.dumps({'step': 'error', 'message': '音频解码失败: 无帧数据'})}\n\n"
                    return

                sample_rate = all_frames[0].sample_rate
                samples_per_segment = SEGMENT_DURATION * sample_rate

                # Split frames into segments by sample count
                current_segment = 0
                current_samples = 0
                segment_frames = []

                def write_segment(frames, seg_idx):
                    seg_path = _new_temp_path(f"_seg{seg_idx}.wav")
                    out = pyav.open(seg_path, 'w', format='wav')
                    out_stream = out.add_stream('pcm_s16le', rate=16000, layout='mono')
                    resampler = pyav.AudioResampler(format='s16', layout='mono', rate=16000)
                    for fr in frames:
                        fr.pts = None
                        for resampled in resampler.resample(fr):
                            for pkt in out_stream.encode(resampled):
                                out.mux(pkt)
                    for pkt in out_stream.encode():
                        out.mux(pkt)
                    out.close()
                    return seg_path

                for frame in all_frames:
                    segment_frames.append(frame)
                    current_samples += frame.samples
                    if current_samples >= samples_per_segment:
                        seg_path = write_segment(segment_frames, current_segment)
                        chunk_paths.append(seg_path)
                        current_segment += 1
                        current_samples = 0
                        segment_frames = []

                # Write remaining frames
                if segment_frames:
                    seg_path = write_segment(segment_frames, current_segment)
                    chunk_paths.append(seg_path)

            except Exception as conv_err:
                yield f"data: {json.dumps({'step': 'error', 'message': f'音频转换失败: {conv_err}'})}\n\n"
                return
            finally:
                if os.path.exists(m4s_path):
                    os.unlink(m4s_path)

            num_chunks = len(chunk_paths)
            total_dur_min = (num_chunks * SEGMENT_DURATION) / 60
            yield f"data: {json.dumps({'step': 'asr', 'message': f'共 {num_chunks} 段 (~{total_dur_min:.0f}分钟)，5路并发识别中...'})}\n\n"

            # Transcribe segments concurrently (5 at a time)
            CONCURRENCY = 5
            semaphore = asyncio.Semaphore(CONCURRENCY)
            results = [None] * num_chunks  # preserve order
            completed = [0]  # mutable counter

            async def transcribe_one(idx, path):
                async with semaphore:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            form = aiohttp.FormData()
                            form.add_field('model', 'glm-asr-2512')
                            form.add_field('stream', 'false')
                            with open(path, 'rb') as audio_file:
                                form.add_field(
                                    'file',
                                    audio_file,
                                    filename=f'seg{idx}.wav',
                                    content_type='audio/wav',
                                )
                                async with aiohttp.ClientSession() as session:
                                    async with session.post(
                                        asr_endpoint,
                                        data=form,
                                        headers={"Authorization": f"Bearer {api_key}"},
                                        timeout=aiohttp.ClientTimeout(total=120),
                                    ) as resp:
                                        resp_text = await resp.text()
                                        completed[0] += 1
                                        if resp.status == 200:
                                            chunk_result = json.loads(resp_text)
                                            text = chunk_result.get('text', '')
                                            if text:
                                                results[idx] = text
                                            print(f"[ASR seg {idx}] OK ({completed[0]}/{num_chunks})")
                                            return
                                        elif resp.status == 429 or resp.status >= 500:
                                            wait = 2 ** attempt
                                            print(f"[ASR seg {idx}] Retry {attempt+1}/{max_retries} after {wait}s: HTTP {resp.status}")
                                            await asyncio.sleep(wait)
                                        else:
                                            print(f"[ASR seg {idx}] FAILED: {resp_text[:200]}")
                                            return
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            wait = 2 ** attempt
                            print(f"[ASR seg {idx}] Network error, retry {attempt+1}/{max_retries} after {wait}s: {e}")
                            await asyncio.sleep(wait)
                    print(f"[ASR seg {idx}] FAILED after {max_retries} retries")

            try:
                tasks = [transcribe_one(i, p) for i, p in enumerate(chunk_paths)]
                batch_size = CONCURRENCY
                for batch_start in range(0, len(tasks), batch_size):
                    batch = tasks[batch_start:batch_start + batch_size]
                    await asyncio.gather(*batch)
                    done_count = min(batch_start + batch_size, num_chunks)
                    yield f"data: {json.dumps({'step': 'asr', 'message': f'转录进度: {done_count}/{num_chunks}'})}\n\n"
            finally:
                for p in chunk_paths:
                    if os.path.exists(p):
                        os.unlink(p)

            transcript = ' '.join(t for t in results if t)
            if not transcript:
                yield f"data: {json.dumps({'step': 'error', 'message': 'ASR 返回空文本'})}\n\n"
                return

            transcript_len = len(transcript)
            yield f"data: {json.dumps({'step': 'transcribed', 'message': f'转录完成 ({transcript_len} 字)'})}\n\n"

            # Step 5: LLM Summarization
            yield f"data: {json.dumps({'step': 'summarize', 'message': '生成总结中...'})}\n\n"
            summary_text, llm_time = await summarize_with_claude(
                subtitle=transcript, title=title, client=ai_client, model=DEFAULT_MODEL
            )

            # Step 6: Save result
            nosub_path = DATA_DIR / "summary" / output_subdir / "no_subtitle" / f"{safe_title}.md"
            if nosub_path.exists():
                nosub_path.unlink()

            save_summary(
                title=title, bvid=bvid, url=url, duration=duration,
                summary=summary_text, output_subdir=output_subdir,
                author_name=author_name, author_uid=author_uid,
                cover_url=cover_url,
            )

            new_path = f"{output_subdir}/{safe_title}.md"
            yield f"data: {json.dumps({'step': 'done', 'message': '总结完成!', 'path': new_path, 'llm_time': round(llm_time, 1)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
