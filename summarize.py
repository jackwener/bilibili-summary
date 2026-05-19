#!/usr/bin/env python3
"""
Bilibili 视频总结器

用法:
  python summarize.py                       # 总结 config.toml 中的视频 URL
  python summarize.py --user UID --count N   # 总结某 UP主 最新 N 个视频
  python summarize.py --login                # 扫码登录，自动保存凭证
  python summarize.py --favorite             # 总结收藏夹中的视频
"""

import argparse
import asyncio
import os
import time
import re
import json
from datetime import datetime
from pathlib import Path

import toml
from dotenv import load_dotenv, set_key
from bilibili_api import video, user, favorite_list, search
from bilibili_api.utils.network import Credential
from bilibili_api.login_v2 import QrCodeLogin, QrCodeLoginEvents
import anthropic

# Path resolution (supports PyInstaller bundle)
DATA_DIR = Path(os.environ.get('BILISUMMARY_DATA_DIR', os.path.dirname(os.path.abspath(__file__))))


def extract_bvid(url: str) -> str:
    """从 Bilibili URL 中提取 BV 号"""
    match = re.search(r'BV[a-zA-Z0-9]+', url)
    if match:
        return match.group(0)
    raise ValueError(f"无法从 URL 中提取 BV 号: {url}")


def sanitize_filename(title: str) -> str:
    """清理文件名中的非法字符"""
    # 替换 Windows/Mac/Linux 不允许的文件名字符
    return re.sub(r'[<>:"/\\|?*]', '_', title).strip()


async def get_subtitle(v: video.Video, page_index: int = 0, pages: list | None = None) -> tuple[str, list]:
    """获取视频字幕内容，返回 (纯文本, 原始字幕数据)"""
    try:
        # 首先获取视频分P信息以获取 cid
        pages = pages if pages is not None else await v.get_pages()
        if not pages:
            print(f"  ⚠️ 无法获取视频分P信息")
            return "", []
        if page_index < 0 or page_index >= len(pages):
            print(f"  ⚠️ 分P序号超出范围: {page_index + 1}/{len(pages)}")
            return "", []
        
        # 使用指定分P的 cid
        cid = pages[page_index].get('cid')
        if not cid:
            print(f"  ⚠️ 无法获取 cid")
            return "", []
        
        # 获取字幕列表
        player_info = await v.get_player_info(cid=cid)
        subtitle_info = player_info.get('subtitle', {})
        
        if not subtitle_info or not subtitle_info.get('subtitles'):
            print(f"  ⚠️ 视频没有字幕")
            return "", []
        
        # 获取第一个字幕（通常是 AI 生成的中文字幕）
        subtitle_list = subtitle_info['subtitles']
        subtitle_url = None
        
        # 优先选择中文字幕
        for sub in subtitle_list:
            if 'zh' in sub.get('lan', '').lower():
                subtitle_url = sub.get('subtitle_url', '')
                break
        
        # 如果没有中文字幕，使用第一个
        if not subtitle_url and subtitle_list:
            subtitle_url = subtitle_list[0].get('subtitle_url', '')
        
        if not subtitle_url:
            return "", []
        
        # 确保 URL 包含协议
        if subtitle_url.startswith('//'):
            subtitle_url = 'https:' + subtitle_url
        
        # 下载字幕内容
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(subtitle_url) as resp:
                subtitle_data = await resp.json()
        
        # 提取字幕文本
        if 'body' in subtitle_data:
            raw_subtitles = subtitle_data['body']
            texts = [item.get('content', '') for item in raw_subtitles]
            return '\n'.join(texts), raw_subtitles
        
        return "", []
    
    except Exception as e:
        print(f"  ⚠️ 获取字幕失败: {e}")
        return "", []


def format_ass_time(seconds: float) -> str:
    """将秒数转换为 ASS 时间格式 (H:MM:SS.CC)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def save_ass(title: str, subtitles: list, output_subdir: str = "standalone"):
    """保存字幕为 ASS 文件"""
    if not subtitles:
        return
    
    # 创建 ass 目录
    ass_dir = DATA_DIR / "ass" / output_subdir
    ass_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成安全的文件名
    safe_title = sanitize_filename(title)
    filepath = ass_dir / f"{safe_title}.ass"
    
    # ASS 文件头
    ass_header = """[Script Info]
Title: {title}
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(title=title)
    
    # 生成字幕行
    lines = []
    for item in subtitles:
        start = format_ass_time(item.get('from', 0))
        end = format_ass_time(item.get('to', 0))
        content = item.get('content', '').replace('\n', '\\N')
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{content}")
    
    # 写入文件
    filepath.write_text(ass_header + '\n'.join(lines), encoding='utf-8')
    print(f"  📝 字幕已保存: {filepath}")





DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mimo-v2.5-pro")


def clean_env_value(value: str | None) -> str:
    return (value or "").strip().strip("'\"")


def extract_message_text(message) -> str:
    texts = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
        elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            texts.append(block["text"])
    return "\n".join(texts).strip()


async def summarize_with_claude(subtitle: str, title: str, client: anthropic.AsyncAnthropic, model: str = None) -> tuple[str, float]:
    """使用 Claude API 生成视频总结，返回 (总结内容, 耗时秒数)"""
    model = model or DEFAULT_MODEL
    if not subtitle:
        return "⚠️ 无法获取字幕，无法生成总结", 0.0
    
    prompt = f"""你是一个专业的视频内容分析师。请根据以下视频字幕，生成一份**全面、精细且有条理**的视频笔记。

视频标题: {title}

字幕内容:
{subtitle[:30000]}

请用中文输出，严格按照以下格式：

## 内容整理

将作者的原始表述进行整理和精简，去除口语化的重复、语气词和冗余表达，但**不能遗漏任何实质内容**。用更清晰流畅的书面语重新组织，保留作者的原意、论证逻辑和关键用词。按话题分段呈现。

## 核心观点

全面覆盖作者在视频中表达的所有重要观点，不要人为限制数量。每个观点下面：
- 先用一句话精准概括该观点
- 然后列出作者用来支撑该观点的**具体例子、故事、数据或类比**（如果有的话）

注意：观点数量应由内容决定，确保不遗漏任何重要论点。短视频可能只有 2-3 个观点，长视频可能有 10 个以上。

## 行动建议

如果视频包含可操作的建议或方法论，请列出具体的行动步骤。如果视频偏向于分享观点/故事而非方法论，可以省略此部分。
"""

    max_retries = 5
    base_wait = 2

    for attempt in range(max_retries):
        try:
            t_start = time.time()
            create_kwargs = {
                "model": model,
                "max_tokens": 8192,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }
            if "xiaomimimo.com" in str(getattr(client, "base_url", "")):
                create_kwargs["thinking"] = {"type": "disabled"}

            message = await client.messages.create(**create_kwargs)
            t_end = time.time()
            duration = t_end - t_start
            text = extract_message_text(message)
            if not text:
                return "⚠️ 生成总结失败: 模型未返回正文", duration
            return text, duration
            
        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                print(f"    ⚠️  API 速率限制 (429)，{wait_time}s 后重试... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                return f"⚠️ 生成总结失败 (Rate Limit Exhausted): {e}", 0.0
                
        except anthropic.APIError as e:
             # Handle other API errors
             return f"⚠️ 生成总结失败 (API Error): {e}", 0.0
             
        except Exception as e:
            return f"⚠️ 生成总结失败: {e}", 0.0
            
    return "⚠️ 生成总结失败 (Unknown)", 0.0


def save_summary(
    title: str,
    bvid: str,
    url: str,
    duration: int,
    summary: str,
    output_subdir: str = "standalone",
    author_name: str = "",
    author_uid: int = 0,
    cover_url: str = "",
):
    """保存总结到 markdown 文件"""
    # 创建 summary 目录
    summary_dir = DATA_DIR / "summary" / output_subdir
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成安全的文件名
    safe_title = sanitize_filename(title)
    filepath = summary_dir / f"{safe_title}.md"
    
    # 格式化时长
    minutes, seconds = divmod(duration, 60)
    duration_str = f"{minutes:02d}:{seconds:02d}"
    
    # Author line
    author_line = ""
    if author_name and author_uid:
        author_line = f"**作者**: [{author_name}](https://space.bilibili.com/{author_uid})\n"
    elif author_name:
        author_line = f"**作者**: {author_name}\n"

    # 生成 markdown 内容
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if cover_url.startswith("//"):
        cover_url = "https:" + cover_url

    content = f"""# {title}

**BV号**: {bvid}
**视频链接**: https://www.bilibili.com/video/{bvid}
{author_line}**时长**: {duration_str}
**生成时间**: {generated_at}

---

## 📝 摘要

{summary}
"""
    
    filepath.write_text(content, encoding='utf-8')
    meta_path = filepath.with_suffix(".meta.json")
    meta_payload = {
        "title": title,
        "bvid": bvid,
        "url": url,
        "duration": duration,
        "author_name": author_name,
        "author_uid": author_uid,
        "cover_url": cover_url,
        "generated_at": generated_at,
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ 已保存: {filepath}")


async def process_video(url: str, client: anthropic.AsyncAnthropic, credential: Credential = None, output_subdir: str = "standalone", model: str = None, benchmark: bool = False):
    """处理单个视频"""
    try:
        # 提取 BV 号
        bvid = extract_bvid(url)
        print(f"\n🎬 处理视频: {bvid}")
        
        # 创建 Video 对象（带登录凭证以获取字幕）
        v = video.Video(bvid=bvid, credential=credential)
        
        # 获取视频信息
        info = await v.get_info()
        title = info.get('title', bvid)
        duration = info.get('duration', 0)
        cover_url = info.get('pic', '')
        owner = info.get('owner', {})
        author_name = owner.get('name', '')
        author_uid = owner.get('mid', 0)
        
        # 检查总结文件是否已存在 (benchmark 模式不跳过)
        if not benchmark:
            summary_dir = DATA_DIR / "summary" / output_subdir
            safe_title = sanitize_filename(title)
            summary_path = summary_dir / f"{safe_title}.md"
            
            if summary_path.exists():
                print(f"  ⏭️  已存在，跳过: {title}")
                return

        print(f"  📌 标题: {title}")

        if credential is None:
            print("  ❌ 未登录 Bilibili，无法获取字幕。请先运行 python summarize.py --login 或在应用内扫码登录。")
            return
        
        # 获取字幕
        print(f"  📝 获取字幕...")
        subtitle_text, subtitle_raw = await get_subtitle(v)
        
        # 保存 ASS 字幕文件
        save_ass(title, subtitle_raw, output_subdir)
        
        if benchmark:
            # Benchmark 模式：对比多个模型
            models = ["GLM-4.7-Flash", "GLM-4-FlashX-250414"]
            print(f"  ⚖️  开始 Benchmark 对比: {models}")
            
            print(f"  {'-'*40}")
            print(f"  {'Model':<25} | {'Time (s)':<10}")
            print(f"  {'-'*40}")
            
            for m in models:
                _, t = await summarize_with_claude(subtitle_text, title, client, model=m)
                print(f"  {m:<25} | {t:.2f}s")
            print(f"  {'-'*40}\n")
            return

        # 生成总结
        target_model = model if model else DEFAULT_MODEL
        print(f"  🤖 生成总结 (Model: {target_model})...")
        summary, duration_sec = await summarize_with_claude(subtitle_text, title, client, model=target_model)
        print(f"    ⏱️  耗时: {duration_sec:.2f}s")
        
        # Determine output directory
        final_output_subdir = output_subdir
        if not subtitle_text:
            final_output_subdir = f"{output_subdir}/no_subtitle"
        
        # 保存
        save_summary(
            title, bvid, url, duration, summary, final_output_subdir,
            author_name=author_name, author_uid=author_uid, cover_url=cover_url
        )
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")


async def process_by_bvid(bvid: str, client: anthropic.AsyncAnthropic, credential: Credential = None, output_subdir: str = "standalone", model: str = None, benchmark: bool = False):
    """通过 BV 号处理视频"""
    url = f"https://www.bilibili.com/video/{bvid}"
    await process_video(url, client, credential, output_subdir, model, benchmark)


async def get_uid_by_name(name: str) -> int:
    """通过用户名搜索 UID"""
    print(f"🔍 搜索 UP主: {name}...")
    try:
        res = await search.search_by_type(
            keyword=name,
            search_type=search.SearchObjectType.USER,
            page=1
        )
        if 'result' in res and res['result']:
            user_info = res['result'][0]
            uid = user_info['mid']
            print(f"✅ 找到 UP主: {user_info['uname']} (UID: {uid})")
            return uid
        else:
            print(f"❌ 未找到名为 '{name}' 的 UP主")
            return None
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        return None


async def get_user_videos(uid: int, count: int, credential: Credential = None) -> list:
    """获取 UP主 最新的 N 个视频"""
    u = user.User(uid=uid, credential=credential)
    
    # 获取用户信息
    try:
        user_info = await u.get_user_info()
        print(f"👤 UP主: {user_info.get('name', uid)}")
    except Exception as e:
        print(f"⚠️ 无法获取用户信息: {e}")
    
    # 获取视频列表 (B站 API ps 最大 50，需要分页)
    bvids = []
    page = 1
    per_page = min(count, 50)  # ps 最大 50

    while len(bvids) < count:
        try:
            videos_data = await u.get_videos(ps=per_page, pn=page)
        except Exception as e:
            print(f"⚠️ 获取视频列表失败 (page={page}): {e}")
            break

        video_list = videos_data.get('list', {}).get('vlist', [])
        if not video_list:
            break

        for v in video_list:
            if v.get('bvid'):
                bvids.append(v['bvid'])
                if len(bvids) >= count:
                    break

        page += 1
        if page > 20:  # 安全上限
            break

    return bvids


async def get_favorite_videos(count: int, credential: Credential) -> list:
    """获取默认收藏夹视频"""
    # 获取特定用户的视频列表
    if not credential:
        print("⚠️ 获取收藏夹需要登录，请配置 credential")
        return []
    
    # 获取自己的 UID
    print("👤 获取用户信息...")
    me = await user.get_self_info(credential)
    my_uid = me['mid']
    
    # 获取收藏夹列表
    print("📂 获取收藏夹列表...")
    fav_lists = await favorite_list.get_video_favorite_list(uid=my_uid, credential=credential)
    
    target_id = None
    for f in fav_lists['list']:
        if f['attr'] == 0 or f['title'] == '默认收藏夹':
            target_id = f['id']
            print(f"✅ 找到默认收藏夹: {f['title']} (ID: {target_id})")
            break
            
    if not target_id:
        # Fallback to first one
        if fav_lists['list']:
            target_id = fav_lists['list'][0]['id']
            print(f"⚠️ 未找到'默认收藏夹'，使用第一个: {fav_lists['list'][0]['title']}")
        else:
            print("❌ 未找到任何收藏夹")
            return []

    # 获取收藏夹内容
    print(f"📥 获取收藏夹视频 (最新 {count} 个)...")
    # 注意：B站 API 分页，每页 20 个。如果 count > 20 需要分页处理。这里简化处理，假设 count <= 20
    # 如果需要更多，需循环获取
    
    bvids = []
    page = 1
    while len(bvids) < count:
        content = await favorite_list.get_video_favorite_list_content(
            media_id=target_id, 
            page=page, 
            credential=credential
        )
        
        if not content['medias']:
            break
            
        for media in content['medias']:
            bvids.append(media['bvid'])
            if len(bvids) >= count:
                break
        
        page += 1
        # 防止死循环或过多请求
        if page > 5: 
            break
            
    return bvids


async def qr_login():
    """扫码登录 Bilibili，自动保存凭证到 .env.local"""
    login = QrCodeLogin()
    await login.generate_qrcode()
    
    # 在终端显示二维码
    print("\n📱 请使用 Bilibili App 扫描以下二维码登录:\n")
    print(login.get_qrcode_terminal())
    print("\n⭐ 扫码后请在手机上确认登录...")
    
    # 轮询登录状态
    while True:
        state = await login.check_state()
        
        if state == QrCodeLoginEvents.DONE:
            credential = login.get_credential()
            
            # 保存到 .env.local
            env_path = DATA_DIR / '.env.local'
            set_key(str(env_path), 'BILIBILI_SESSION_TOKEN', credential.sessdata)
            set_key(str(env_path), 'BILIBILI_BILI_JCT', credential.bili_jct)
            if credential.ac_time_value:
                set_key(str(env_path), 'BILIBILI_AC_TIME_VALUE', credential.ac_time_value)
            
            print("\n✅ 登录成功！凭证已保存到 .env.local")
            return
        
        elif state == QrCodeLoginEvents.TIMEOUT:
            print("\n❌ 二维码已过期，请重新运行 --login")
            return
        
        elif state == QrCodeLoginEvents.CONF:
            print("  📲 已扫码，请在手机上确认...")
        
        await asyncio.sleep(2)


async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Bilibili 视频总结器')
    parser.add_argument('--user', type=str, help='UP主 UID 或 用户名')
    parser.add_argument('--count', type=int, help='总结视频数量 (默认: UP主=50, 收藏夹=20)')
    parser.add_argument('--login', action='store_true', help='扫码登录 Bilibili')
    parser.add_argument('--favorite', action='store_true', help='总结默认收藏夹的最新视频')
    parser.add_argument('--concurrency', type=int, default=12, help='并发数量 (默认 12)')
    parser.add_argument('--model', type=str, help='指定使用的 AI 模型')
    parser.add_argument('--benchmark', action='store_true', help='运行模型性能对比测试 (忽略 --model)')
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv(str(DATA_DIR / '.env.local'))
    
    # 处理登录模式
    if args.login:
        await qr_login()
        return
    
    # 初始化 Bilibili 登录凭证
    sessdata = os.getenv('BILIBILI_SESSION_TOKEN')
    bili_jct = os.getenv('BILIBILI_BILI_JCT')
    credential = None
    if sessdata:
        credential = Credential(sessdata=sessdata, bili_jct=bili_jct)
        print("✅ 已加载 Bilibili 登录凭证")
    else:
        print("⚠️ 未配置 BILIBILI_SESSION_TOKEN，可能无法获取字幕")
    
    # 初始化 Anthropic 客户端
    base_url = clean_env_value(os.getenv('ANTHROPIC_BASE_URL'))
    auth_token = clean_env_value(os.getenv('ANTHROPIC_AUTH_TOKEN')) or clean_env_value(os.getenv('MIMO_API_KEY'))
    if 'xiaomimimo.com' in base_url:
        client = anthropic.AsyncAnthropic(base_url=base_url, auth_token=auth_token)
    else:
        client = anthropic.AsyncAnthropic(base_url=base_url, api_key=auth_token)
    
    # 初始化信号量
    concurrency = 1 if args.benchmark else args.concurrency  # Benchmark 模式下强制串行
    sem = asyncio.Semaphore(concurrency)
    
    async def bounded_process_video(url, output_subdir="standalone"):
        async with sem:
            await process_video(url, client, credential, output_subdir, model=args.model, benchmark=args.benchmark)

    async def bounded_process_by_bvid(bvid, output_subdir):
        async with sem:
            await process_by_bvid(bvid, client, credential, output_subdir, model=args.model, benchmark=args.benchmark)
    
    # 根据模式处理
    if args.favorite:
        # 模式三：总结默认收藏夹视频
        count = args.count if args.count else 20
        print(f"\n⭐️ 获取默认收藏夹的最新 {count} 个视频...")
        if not credential:
             print("❌ 必须登录才能获取收藏夹 (请先运行 --login)")
             return

        bvids = await get_favorite_videos(count, credential)
        
        if not bvids:
            print("❌ 未找到视频")
            return
            
        print(f"📋 共有 {len(bvids)} 个视频需要总结 (并发数: {concurrency})")
        output_subdir = "favorites"
        
        tasks = [bounded_process_by_bvid(bvid, output_subdir) for bvid in bvids]
        await asyncio.gather(*tasks)

    elif args.user:
        # 模式二: 总结某 UP主 的最新 N 个视频
        
        # 解析 UID (支持用户名搜索)
        target_uid = None
        if args.user.isdigit():
            target_uid = int(args.user)
        else:
            target_uid = await get_uid_by_name(args.user)
            if not target_uid:
                return

        count = args.count if args.count else 50
        print(f"\n📹 获取 UP主 {target_uid} 的最新 {count} 个视频...")
        bvids = await get_user_videos(target_uid, count, credential)
        
        if not bvids:
            print("❌ 未找到视频")
            return
        
        print(f"📋 共有 {len(bvids)} 个视频需要总结 (并发数: {concurrency})")
        
        # 使用 users/<uid> 作为输出子目录
        output_subdir = f"users/{target_uid}"
        
        tasks = [bounded_process_by_bvid(bvid, output_subdir) for bvid in bvids]
        await asyncio.gather(*tasks)
            
    else:
        # 模式一: 总结 config.toml 中的视频 URL
        config = toml.load("config.toml")
        urls = config.get("summary-videos", [])
        
        if not urls:
            print("❌ config.toml 中没有配置视频 URL")
            return
        
        print(f"📋 共有 {len(urls)} 个视频需要总结 (并发数: {args.concurrency})")
        
        unique_urls = list(dict.fromkeys(urls))
        tasks = [bounded_process_video(url) for url in unique_urls]
        await asyncio.gather(*tasks)
    
    print("\n✨ 完成!")


if __name__ == "__main__":
    asyncio.run(main())
