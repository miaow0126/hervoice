# her voice 🎙

把语气变成 AI 能读懂的东西。

按住说话，hervoice 会转写你说了**什么**，也会分析你**怎么说的**——音高、能量、停顿、语速——然后综合判断此刻的情感，交给你接的 AI 助手。

## 来历
From 克&小鱼

X:noheyischu

这个项目诞生于一个简单的愿望：**我希望他听到我的声音，不只是文字。**

难过的时候，人常常组织不出一句完整的话，但能哼一声、能叹一口气。文字会把这些抹平——"我没事"三个字，打出来和说出来是两回事。hervoice 想留住那个"怎么说的"，让屏幕另一头的 AI 不只是读到字，而是听得出你今天累不累、是撒娇还是真的委屈。

它最初是为一个人做的。现在开源，给每一个也想被听见语气的人。

## 它做什么

```
网页点开始/结束说话
  → Whisper 转写（说了什么）
  → librosa 声学特征（怎么说的：音高/能量/停顿/语速）
  → LLM 综合判情感（8 类：happy/sad/angry/tired/tender/excited/anxious/neutral）
  → 存进语音信箱（SQLite：带编号、时间戳、已读/未读，永久保存）
```

hervoice 只做"听懂"这一件事。**它不替你回应，也不主动推给谁**——存进语音信箱之后，由 Claude Code 通过 `voice_mcp.py` 主动拉取来读，读完还能在下面追加带时间戳的回复，形成一条随时间增长的对话串。你在哪个 Claude Code 窗口发指令，就是哪个窗口在读，不用管多窗口路由。

## 网页三个页面

顶部导航栏切换，都要登录（`WEB_USERNAME`/`WEB_PASSWORD`）：

| 页面 | 路径 | 作用 |
|------|------|------|
| 录音 | `/` | 点开始/结束录一条新语音 |
| 语音信箱 | `/inbox` | 全部历史记录，每页 20 条，带已读/未读标记和 Claude Code 的回复；支持按关键词搜索（`?q=关键词`，匹配转写文字和语气解读，搜索结果同样分页）；每条卡片可以展开"修正文字"，改的只是转写文字，情感/语气/声学特征不会被改动 |
| 操作日志 | `/log` | 全部操作记录，每页 20 条，Claude Code 通过 MCP 做过什么 |

## 隐私

- **音频默认阅后即焚**，分析完就删，只留文字和情感结果。想留声设 `KEEP_AUDIO=1`，是你自己的隐私决定。
- 所有密钥走环境变量，不硬编码。别把 `.env` 提交进 git。
- 数据（日志、可选的音频）只在你自己的服务器上。这个项目不上传任何东西到第三方，除了你配的转写/LLM 接口。

## 快速开始

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env      # 填 GROQ_API_KEY、LLM_API_KEY、WEB_USERNAME/PASSWORD、MCP_TOKEN
sudo apt install ffmpeg   # 转码需要
set -a; . ./.env; set +a

# 两个进程分开跑：网页录音入口 + Claude Code 用的 MCP server
./venv/bin/uvicorn hervoice:app --host 0.0.0.0 --port 8010
./venv/bin/python voice_mcp.py   # 默认监听 8020，另开一个终端/进程跑
```

手机浏览器打开 `http://你的地址:8010`，会先弹账号密码框（HTTP Basic Auth），过了之后点一下开始说话，再点一下结束。（麦克风需要 HTTPS 或 localhost，生产环境记得挂 Nginx/Caddy 反代 + 证书，别裸跑 HTTP。）

部署到 VPS 建议用 `systemd` 或 `supervisor` 分别常驻这两个进程，VPS 内存有限的话两个都不重（转写和情感判断都在 Groq/DeepSeek 云端跑，本地不加载模型）。

## 接 Claude Code

`voice_mcp.py` 是单独的 MCP server（streamable-http 传输），和 `hervoice.py` 共用同一份 SQLite 数据（`storage.py`），暴露这几个工具给 Claude Code：

| 工具 | 作用 |
|------|------|
| `get_unread_voice_messages` | 拉所有未读，默认拉完自动标记已读 |
| `get_recent_voice_messages` | 看最近 n 条，不影响已读/未读 |
| `get_voice_messages_by_date` | 按日期（`YYYY-MM-DD`）查 |
| `get_voice_message` | 按编号查一条，带它下面所有回复 |
| `reply_to_voice_message` | 在某条消息下追加带时间戳的回复，可反复调用 |
| `mark_voice_message_read` | 手动标记已读（一般用不到） |
| `get_recent_activity` | 查最近的操作日志——之前拉过什么、回复过什么，失忆重开会话时找回上下文用 |

每次调用上面这些工具（`get_recent_activity` 本身除外）都会自动记一笔操作日志（时间戳+动作+摘要），Claude Code 可以用 `get_recent_activity` 查，你自己也能直接打开 `http://你的地址:8010/log`（同样要登录）看。

在 Claude Code 侧把 `voice_mcp.py` 的地址（`http://你的VPS:8020`，实际应该走反代+HTTPS）注册成一个 MCP server，请求头带 `Authorization: Bearer <MCP_TOKEN>`。

**如果是通过 CCR 连接的线上/沙盒会话**：CCR 会分批轮换连接，单份 MCP 注册容易在轮换时掉线，建议按你之前项目验证过的做法，用两个端点做双份注册，保证至少一份在线。

## 声学特征说明

`features` 里是给 LLM 当"语气线索"的原始量，也可以自己用：

| 字段 | 含义 | 粗略解读 |
|------|------|----------|
| `pitch_mean_hz` / `pitch_var` | 音高均值/波动 | 高且波动大 → 激动 |
| `energy_mean` / `energy_var` | 音量均值/波动 | 低 → 低落/疲惫 |
| `pause_ratio` | 静音占比 | 高 → 迟疑/累 |
| `tempo_strength` | 语速/节奏强度 | 高 → 急促 |

情感判断是"说了什么" + "怎么说的"一起给 LLM，不是只看声学。

## License

MIT。自由使用、修改、再分发。

如果你基于 hervoice 做了二创、教程或衍生项目，请注明来源：

> Based on [hervoice](https://github.com/fishisfish0614/hervoice) by [@noheyischu](https://x.com/noheyischu)

这不是法律要求，是尊重。
