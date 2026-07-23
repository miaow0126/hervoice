#!/usr/bin/env python3
"""hervoice · 把语气变成 AI 能读懂的东西

录音 → Whisper 转写 + librosa 声学特征 → LLM 综合判情感 → 存进语音信箱

语音信箱：每条消息永久保存、带编号和已读/未读状态，Claude Code 通过
voice_mcp.py 主动拉取/回复，不依赖谁推送给哪个窗口。

配置见 .env（复制 .env.example）。隐私默认：音频阅后即焚，KEEP_AUDIO=1 才留存。
网页录音入口需要账号密码（WEB_USERNAME/WEB_PASSWORD）。
"""
import html
import json
import math
import os
import secrets
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import storage

load_dotenv()

# ── 配置（全部走环境变量，不硬编码任何密钥）──
DATA_DIR = Path(os.environ.get("HERVOICE_DATA", "./data"))
CLIPS = DATA_DIR / "clips"
KEEP_AUDIO = os.environ.get("KEEP_AUDIO", "0") == "1"   # 默认阅后即焚

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "zh")
# 低于这个平均能量就算"没怎么说话"，跳过转写，避免 Whisper 在安静片段上瞎编字幕
SILENCE_ENERGY_THRESHOLD = float(os.environ.get("SILENCE_ENERGY_THRESHOLD", "0.006"))

WEB_USERNAME = os.environ.get("WEB_USERNAME", "")
WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "")

EMOTIONS = ["happy", "sad", "angry", "tired", "tender", "excited", "anxious", "neutral"]

NAV_ITEMS = [("/", "录音"), ("/inbox", "语音信箱"), ("/log", "操作日志")]


def _nav(active: str) -> str:
    links = "".join(
        f'<a href="{href}"{" class=active" if href == active else ""}>{label}</a>'
        for href, label in NAV_ITEMS)
    return f"<nav>{links}</nav>"


NAV_STYLE = """
nav{display:flex;gap:18px;font-size:.78rem;letter-spacing:.08em}
nav a{color:var(--fg);opacity:.5;text-decoration:none;padding-bottom:3px;border-bottom:2px solid transparent}
nav a:hover{opacity:.8}
nav a.active{opacity:1;border-color:var(--accent)}
nav a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

PAGER_STYLE = """
.pager{display:flex;gap:16px;align-items:center;font-size:.78rem;letter-spacing:.05em;margin-top:4px}
.pager a{color:var(--accent);text-decoration:none}
.pager a:hover{opacity:.7}
.pager a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.pager .disabled{opacity:.3}
"""


def _pager(page: int, page_size: int, total: int, base: str, extra_qs: str = "") -> str:
    total_pages = max(1, math.ceil(total / page_size))
    page = min(max(page, 1), total_pages)
    qs = f"&{extra_qs}" if extra_qs else ""
    if page > 1:
        prev = f'<a href="{base}?page={page - 1}{qs}">‹ 上一页</a>'
    else:
        prev = '<span class=disabled>‹ 上一页</span>'
    if page < total_pages:
        nxt = f'<a href="{base}?page={page + 1}{qs}">下一页 ›</a>'
    else:
        nxt = '<span class=disabled>下一页 ›</span>'
    return f'<div class=pager>{prev}<span>第 {page} / {total_pages} 页</span>{nxt}</div>'

storage.init(DATA_DIR)
app = FastAPI()
security = HTTPBasic()


def require_login(credentials: HTTPBasicCredentials = Depends(security)):
    if not WEB_USERNAME or not WEB_PASSWORD:
        raise HTTPException(500, "WEB_USERNAME/WEB_PASSWORD 未配置，网页入口已锁死")
    ok = (secrets.compare_digest(credentials.username, WEB_USERNAME) and
          secrets.compare_digest(credentials.password, WEB_PASSWORD))
    if not ok:
        raise HTTPException(401, "账号或密码不对", headers={"WWW-Authenticate": "Basic"})
    return True


def _llm(prompt, max_tokens=200):
    body = json.dumps({"model": LLM_MODEL, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{LLM_BASE}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {LLM_KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def _whisper(wav_path):
    if not GROQ_KEY:
        return None, "GROQ_API_KEY not set"
    r = subprocess.run(["curl", "-s", "-m", "60",
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        "-H", f"Authorization: Bearer {GROQ_KEY}",
                        "-F", f"file=@{wav_path}", "-F", f"model={WHISPER_MODEL}",
                        "-F", f"language={WHISPER_LANG}"],
                       capture_output=True, text=True, timeout=70)
    try:
        return json.loads(r.stdout).get("text", "").strip(), None
    except Exception:
        return None, r.stdout[:200]


def _acoustic_features(wav_path):
    """轻量声学特征：音高/能量/停顿/语速感 — 给 LLM 当'怎么说的'线索"""
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    dur = len(y) / sr
    if dur < 0.3:
        return {"duration_s": round(dur, 1)}
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    f0v = f0[(f0 > 60) & (f0 < 500)]
    rms = librosa.feature.rms(y=y)[0]
    silent = float(np.mean(rms < np.percentile(rms, 20) * 1.5))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    return {
        "duration_s": round(dur, 1),
        "pitch_mean_hz": round(float(np.mean(f0v)), 1) if len(f0v) else 0,
        "pitch_var": round(float(np.std(f0v)), 1) if len(f0v) else 0,
        "energy_mean": round(float(np.mean(rms)), 4),
        "energy_var": round(float(np.std(rms)), 4),
        "pause_ratio": round(silent, 2),
        "tempo_strength": round(float(np.mean(onset)), 2),
    }


def _judge_emotion(text, feats):
    prompt = (
        f"分析一段语音的情感。\n说话内容:「{text}」\n"
        f"声学特征: {json.dumps(feats, ensure_ascii=False)}"
        f"(pitch高+var大=激动; energy低+pause多=低落/疲惫; pitch上扬短句=撒娇可能)\n"
        f"综合'说了什么'和'怎么说的'，从{EMOTIONS}中选1个最贴切的，"
        f'只输出JSON: {{"emotion":"...","confidence":0.0到1.0,"hint":"一句话描述此刻状态"}}'
    )
    raw = _llm(prompt)
    s, e = raw.find("{"), raw.rfind("}")
    return json.loads(raw[s:e + 1])


@app.post("/api/voice/upload")
async def upload(file: UploadFile = File(...), _=Depends(require_login)):
    raw = await file.read()
    clip_name = ""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("in" + (Path(file.filename or "a.webm").suffix or ".webm"))
        src.write_bytes(raw)
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(wav)],
                       capture_output=True, timeout=60)
        if not wav.exists():
            return JSONResponse({"error": "audio convert failed"}, status_code=400)
        feats = _acoustic_features(wav)
        # 基本没声音的片段丢给 Whisper 容易"幻觉"出一段听起来很像真话的文字
        # （常见的是"请点赞订阅转发"这种视频结尾套话）——安静就直接跳过转写
        if feats.get("energy_mean", 1) < SILENCE_ENERGY_THRESHOLD:
            return JSONResponse({"error": "没有检测到说话内容，太安静了"}, status_code=400)
        # 转写用响度标准化过的独立副本——声音偏小也能转得准；声学特征（feats）
        # 继续用没改过响度的原始 wav，不然 energy_mean 这类信号会被标准化抹平
        wav_norm = Path(td) / "a_norm.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(wav), "-af", "loudnorm", str(wav_norm)],
                       capture_output=True, timeout=60)
        whisper_input = wav_norm if wav_norm.exists() else wav
        text, err = _whisper(whisper_input)
        if text is None:
            return JSONResponse({"error": f"whisper failed: {err}"}, status_code=502)
        if KEEP_AUDIO:
            try:
                CLIPS.mkdir(parents=True, exist_ok=True)
                from datetime import datetime, timezone
                clip_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".mp3"
                subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-b:a", "64k",
                                str(CLIPS / clip_name)], capture_output=True, timeout=60)
                if not (CLIPS / clip_name).exists():
                    clip_name = ""
            except Exception:
                clip_name = ""
    try:
        emo = _judge_emotion(text, feats)
    except Exception as e:
        print(f"[hervoice] emotion analysis failed: {e!r}")
        emo = {"emotion": "neutral", "confidence": 0.0, "hint": "emotion analysis failed"}
    msg_id = storage.add_message(
        text=text, emotion=emo.get("emotion", "neutral"),
        confidence=emo.get("confidence", 0), hint=emo.get("hint", ""),
        features=feats, audio=clip_name)
    return {"id": msg_id, "text": text, "emotion": emo.get("emotion", "neutral"),
            "confidence": emo.get("confidence", 0), "hint": emo.get("hint", "")}


@app.get("/api/voice/recent")
async def recent(n: int = 10, _=Depends(require_login)):
    return storage.get_recent(n)


@app.get("/api/voice/audio/{name}")
async def audio_clip(name: str, _=Depends(require_login)):
    from fastapi.responses import FileResponse
    safe = Path(name).name
    fp = CLIPS / safe
    if not safe.endswith(".mp3") or not fp.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(fp, media_type="audio/mpeg")


PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,user-scalable=no">
<link rel=preconnect href=https://fonts.gstatic.com crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Love+Light&display=swap" rel=stylesheet>
<title>her voice</title><style>
:root{--bg:#faf7f2;--fg:#3a3532;--accent:#c96f5e;--soft:#e8ded2}
@media(prefers-color-scheme:dark){:root{--bg:#1c1a18;--fg:#e8e2da;--accent:#d98873;--soft:#3a342e}}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--fg);
font-family:Georgia,'Songti SC',serif;min-height:100vh;min-height:100dvh;display:flex;flex-direction:column;
align-items:center;justify-content:flex-start;gap:28px;padding:24px}
h1{font-family:'Love Light',cursive;font-size:2.6rem;font-weight:400;letter-spacing:.03em;margin-top:12vh}
#btn{width:120px;height:120px;border-radius:50%;border:2px solid var(--accent);
background:var(--accent);color:var(--bg);font-size:1rem;font-family:inherit;
touch-action:none;-webkit-user-select:none;user-select:none;cursor:pointer}
#btn.rec{animation:pulse-ring 1.6s ease-in-out infinite}
@keyframes pulse-ring{
 0%,100%{box-shadow:0 0 0 6px color-mix(in srgb,var(--accent) 22%,transparent)}
 50%{box-shadow:0 0 0 20px color-mix(in srgb,var(--accent) 8%,transparent)}
}
@media(prefers-reduced-motion:reduce){
 #btn.rec{animation:none;box-shadow:0 0 0 14px color-mix(in srgb,var(--accent) 14%,transparent)}
}
#out{max-width:420px;width:100%;display:flex;flex-direction:column;gap:10px}
.card{background:var(--soft);border-radius:14px;padding:14px 16px;font-size:.92rem;line-height:1.55}
.emo{color:var(--accent);font-size:.8rem;letter-spacing:.08em}
#tip{font-size:.78rem;opacity:.55;letter-spacing:.05em}
""" + NAV_STYLE + """
</style></head><body>
""" + _nav("/") + """
<h1>Her Voice 🎙</h1>
<button id=btn>点击开始</button>
<div id=tip>点一下开始说话，再点一下结束</div>
<div id=out></div>
<script>
const btn=document.getElementById('btn'),out=document.getElementById('out'),tip=document.getElementById('tip');
let mr,chunks=[],recording=false;
async function start(){
 try{const s=await navigator.mediaDevices.getUserMedia({audio:true});
 chunks=[];mr=new MediaRecorder(s);mr.ondataavailable=e=>chunks.push(e.data);
 mr.onstop=send;mr.start();recording=true;btn.classList.add('rec');btn.textContent='点击结束';
 tip.textContent='录音中…';}
 catch(e){tip.textContent='需要麦克风权限';}
}
function stop(){if(mr&&mr.state!=='inactive'){mr.stop();mr.stream.getTracks().forEach(t=>t.stop());}
 recording=false;btn.classList.remove('rec');btn.textContent='点击开始';}
async function send(){
 const blob=new Blob(chunks,{type:mr.mimeType||'audio/webm'});
 if(blob.size<1000){tip.textContent='太短了，再说一次';return;}
 tip.textContent='分析中…';
 const fd=new FormData();fd.append('file',blob,'voice.webm');
 try{const r=await fetch('/api/voice/upload',{method:'POST',body:fd});
 if(r.status===401){tip.textContent='登录过期，刷新页面重新登录';return;}
 const d=await r.json();
 if(d.error){tip.textContent=d.error.includes('太安静')?d.error:'出错了: '+d.error;return;}
 const c=document.createElement('div');c.className='card';
 const meta=document.createElement('div');meta.className='emo';
 meta.textContent='#'+d.id+' · '+d.emotion+' · '+(d.hint||'');
 const body=document.createElement('div');body.textContent=d.text;
 c.append(meta,body);
 out.prepend(c);tip.textContent='听到了 ♡';}
 catch(e){tip.textContent='网络出错，再试一次';}
}
btn.addEventListener('click',()=>{recording?stop():start();});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index(_=Depends(require_login)):
    return PAGE


PAGE_SIZE = 20


@app.get("/inbox", response_class=HTMLResponse)
async def inbox_page(q: str = "", page: int = 1, _=Depends(require_login)):
    q = q.strip()
    if q:
        msgs, total = storage.search_messages_page(q, page, PAGE_SIZE)
        empty = '<div class=card>没有匹配的记录</div>'
        title = f'语音信箱 · 搜索“{html.escape(q)}”，命中 {total} 条'
        pager = _pager(page, PAGE_SIZE, total, "/inbox", f"q={urllib.parse.quote(q)}")
    else:
        msgs, total = storage.get_messages_page(page, PAGE_SIZE)
        empty = '<div class=card>还没有语音记录</div>'
        title = f"语音信箱（共 {total} 条，全部）"
        pager = _pager(page, PAGE_SIZE, total, "/inbox")
    items = "".join(_render_message_card(m, page, q) for m in msgs) or empty
    return (INBOX_PAGE_TEMPLATE
            .replace("{{TITLE}}", title)
            .replace("{{ITEMS}}", items)
            .replace("{{Q}}", html.escape(q))
            .replace("{{PAGER}}", pager))


def _render_message_card(m: dict, page: int = 1, q: str = "") -> str:
    status = "已读" if m["read"] else "未读"
    status_cls = "read" if m["read"] else "unread"
    replies = "".join(
        f'<div class=reply><span class=rts>{html.escape(r["ts"])}</span>{html.escape(r["text"])}</div>'
        for r in m.get("replies", []))
    edit_form = (
        f'<details class=edit><summary>修正文字</summary>'
        f'<form method=post action=/api/voice/edit>'
        f'<input type=hidden name=id value="{m["id"]}">'
        f'<input type=hidden name=page value="{page}">'
        f'<input type=hidden name=q value="{html.escape(q)}">'
        f'<textarea name=text>{html.escape(m["text"])}</textarea>'
        f'<button type=submit>保存</button>'
        f'</form></details>'
    )
    delete_form = (
        f'<form method=post action=/api/voice/delete class=delete-form '
        f'onsubmit="return confirm(\'确认删除 #{m["id"]} 语音？删掉就找不回来了\')">'
        f'<input type=hidden name=id value="{m["id"]}">'
        f'<input type=hidden name=page value="{page}">'
        f'<input type=hidden name=q value="{html.escape(q)}">'
        f'<button type=submit>删除</button>'
        f'</form>'
    )
    return (
        f'<div class=card>'
        f'<div class=meta>#{m["id"]} · {html.escape(m["emotion"])}'
        f'{" · " + html.escape(m["hint"]) if m.get("hint") else ""}'
        f' · <span class="status {status_cls}">{status}</span>'
        f' · <span class=ts>{html.escape(m["ts"])}</span></div>'
        f'<div class=body>{html.escape(m["text"])}</div>'
        f'<div class=actions>{edit_form}{delete_form}</div>'
        + (f'<div class=replies>{replies}</div>' if replies else '')
        + '</div>'
    )


@app.post("/api/voice/edit")
async def edit_text(id: int = Form(...), text: str = Form(...),
                     page: int = Form(1), q: str = Form(""),
                     _=Depends(require_login)):
    storage.update_text(id, text.strip())
    qs = f"?page={page}" + (f"&q={urllib.parse.quote(q)}" if q else "")
    return RedirectResponse(url=f"/inbox{qs}", status_code=303)


@app.post("/api/voice/delete")
async def delete_message(id: int = Form(...), page: int = Form(1), q: str = Form(""),
                          _=Depends(require_login)):
    storage.delete_message(id)
    qs = f"?page={page}" + (f"&q={urllib.parse.quote(q)}" if q else "")
    return RedirectResponse(url=f"/inbox{qs}", status_code=303)


@app.get("/log", response_class=HTMLResponse)
async def log_page(page: int = 1, _=Depends(require_login)):
    rows, total = storage.get_activity_page(page, PAGE_SIZE)
    items = "".join(
        f'<div class=card><div class=emo>{html.escape(r["ts"])}</div>'
        f'<div>{html.escape(r["action"])} — {html.escape(r["detail"])}</div></div>'
        for r in rows) or '<div class=card>还没有操作记录</div>'
    pager = _pager(page, PAGE_SIZE, total, "/log")
    return (LOG_PAGE_TEMPLATE
            .replace("{{ITEMS}}", items)
            .replace("{{COUNT}}", str(total))
            .replace("{{PAGER}}", pager))


LIST_PAGE_STYLE = """
:root{--bg:#faf7f2;--fg:#3a3532;--accent:#c96f5e;--soft:#e8ded2}
@media(prefers-color-scheme:dark){:root{--bg:#1c1a18;--fg:#e8e2da;--accent:#d98873;--soft:#3a342e}}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--fg);
font-family:Georgia,'Songti SC',serif;min-height:100vh;padding:24px;
display:flex;flex-direction:column;align-items:center;gap:16px}
h1{font-size:1.1rem;font-weight:400;letter-spacing:.1em}
#out{max-width:560px;width:100%;display:flex;flex-direction:column;gap:10px}
.card{background:var(--soft);border-radius:12px;padding:12px 14px;font-size:.86rem;line-height:1.5}
.meta,.emo{color:var(--accent);font-size:.72rem;letter-spacing:.05em;margin-bottom:6px}
.ts{color:var(--fg);opacity:.5}
.status{opacity:.7}
.status.unread{color:var(--accent);opacity:1;font-weight:bold}
.replies{margin-top:8px;padding-left:12px;border-left:2px solid var(--bg);display:flex;flex-direction:column;gap:6px}
.reply{font-size:.82rem;opacity:.85}
.reply .rts{opacity:.5;margin-right:8px;font-size:.72rem}
.actions{display:flex;align-items:flex-start;gap:14px;margin-top:8px}
details.edit{flex:1}
details.edit summary{font-size:.72rem;color:var(--accent);opacity:.75;cursor:pointer;letter-spacing:.05em}
details.edit summary:hover{opacity:1}
details.edit form{display:flex;gap:8px;margin-top:8px}
details.edit textarea{flex:1;font-family:inherit;font-size:.86rem;padding:8px 10px;
border-radius:8px;border:1px solid var(--bg);background:var(--bg);color:var(--fg);
resize:vertical;min-height:2.4em}
details.edit textarea:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
details.edit button{font-family:inherit;font-size:.78rem;padding:6px 14px;border-radius:8px;
border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;align-self:flex-start}
form.delete-form button{font-family:inherit;font-size:.72rem;letter-spacing:.05em;
background:none;border:none;color:var(--fg);opacity:.4;cursor:pointer;padding:0}
form.delete-form button:hover{opacity:.9}
""" + NAV_STYLE + PAGER_STYLE

SEARCH_STYLE = """
form.search{display:flex;gap:8px;width:100%;max-width:560px}
form.search input{flex:1;font-family:inherit;font-size:.86rem;padding:8px 12px;
border-radius:10px;border:1px solid var(--soft);background:var(--bg);color:var(--fg)}
form.search input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
form.search button{font-family:inherit;font-size:.82rem;padding:8px 16px;border-radius:10px;
border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer}
form.search a{align-self:center;font-size:.78rem;opacity:.6;color:var(--fg);text-decoration:none}
"""

INBOX_PAGE_TEMPLATE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>her voice · 语音信箱</title><style>""" + LIST_PAGE_STYLE + SEARCH_STYLE + """
</style></head><body>
""" + _nav("/inbox") + """
<h1>{{TITLE}}</h1>
<form class=search action=/inbox method=get>
<input name=q value="{{Q}}" placeholder="搜关键词（转写文字/语气解读）…">
<button type=submit>搜</button>
<a href=/inbox>清除</a>
</form>
{{PAGER}}
<div id=out>{{ITEMS}}</div>
{{PAGER}}
</body></html>"""

LOG_PAGE_TEMPLATE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>her voice · 操作日志</title><style>""" + LIST_PAGE_STYLE + """
</style></head><body>
""" + _nav("/log") + """
<h1>操作日志（共 {{COUNT}} 条，全部）</h1>
{{PAGER}}
<div id=out>{{ITEMS}}</div>
{{PAGER}}
</body></html>"""
