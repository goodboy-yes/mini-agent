from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


# 当前 Agent 默认把启动目录视为工作区根目录。
# 后续所有相对路径、命令执行目录、上下文提示中的路径都围绕这里展开。
WORKSPACE_ROOT = Path.cwd().resolve()
# Agent Loop 的最大轮数，用来防止模型因为工具调用反复循环而无法退出。
MAX_TURNS = 20
COMMAND_TIMEOUT_SECONDS = 30
PYTHON_TIMEOUT_SECONDS = 30
DOTENV_FILE_NAME = ".env"
TRACE_MAX_CHARS = 4000
TRACE_HTML_MAX_CHARS = 20000
TRACE_DIR_NAME = ".agent-traces"
CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)
CLIENT_DISCONNECT_WINERRORS = {10053, 10054, 10058}
CLIENT_DISCONNECT_ERRNOS = {32, 104}


@dataclass(slots=True)
class ClientSettings:
    """LLM 客户端配置。"""

    api_key: str
    model: str
    base_url: str | None = None


@dataclass(slots=True)
class ToolSpec:
    """工具注册项，同时保存执行函数和 OpenAI tools schema。"""

    function: Callable[..., "ToolExecutionResult"]
    schema: dict[str, Any]


@dataclass(slots=True)
class ToolExecutionResult:
    """工具执行结果，同时包含返回给模型的内容和执行细节。"""

    content: str
    execution: dict[str, Any]


@dataclass(slots=True)
class TraceSettings:
    """控制终端摘要输出和 HTML 轨迹查看器。"""

    console_mode: str = "summary"
    max_chars: int = TRACE_MAX_CHARS
    html_enabled: bool = True
    html_max_chars: int = TRACE_HTML_MAX_CHARS
    trace_dir: Path = WORKSPACE_ROOT / TRACE_DIR_NAME


@dataclass(slots=True)
class LiveReloadBroadcaster:
    """给 HTML 查看器推送“内容已更新”事件。"""

    condition: threading.Condition = field(default_factory=threading.Condition)
    version: int = 0


@dataclass(slots=True)
class TraceRecorder:
    """统一收集轨迹事件，并分发到终端和 HTML 查看器。"""

    settings: TraceSettings
    session_label: str
    latest_html_path: Path
    session_html_path: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    live_reload_url: str | None = None
    live_viewer_url: str | None = None
    live_reload_broadcaster: LiveReloadBroadcaster | None = None
    live_reload_server: Any | None = None


SYSTEM_PROMPT = f"""你是一个运行在本地工作区内的 AI Agent。

你的工作区根目录是：{WORKSPACE_ROOT}

你可以使用以下工具：
1. shell_exec：执行 shell 命令。Windows 环境下使用 PowerShell。
2. file_read：读取工作区中的文件内容。
3. file_write：写入工作区中的文件内容，如目录不存在则自动创建。
4. python_exec：在子进程中执行 Python 代码。

工作原则：
- 先理解用户目标，再决定是否调用工具。
- 当需要与文件系统、命令行或代码交互时，优先使用工具。
- 所有相对路径都相对于工作区根目录。
- file_read 和 file_write 不能访问工作区根目录之外的路径。
- 工具结果会被加入上下文，请基于最新观察继续推理。
- 当任务已经完成时，直接用自然语言回复最终答案，不要再调用工具。
"""


def _normalize_output(stdout: str, stderr: str, returncode: int | None = None) -> str:
    """把命令/脚本执行结果整理成统一文本，便于回填给模型。"""

    chunks: list[str] = []  # 用列表收集各部分输出内容
    if stdout.strip():
        chunks.append(stdout.strip())  # 标准输出不为空时加入
    if stderr.strip():
        chunks.append(f"[stderr]\n{stderr.strip()}")  # 标准错误不为空时加入，加前缀便于区分
    if returncode not in (None, 0):
        chunks.append(f"[exit code: {returncode}]")  # 非零退出码说明执行出错，附加到末尾
    return "\n\n".join(chunks) or "(no output)"  # 用双换行拼接各部分；全部为空则返回占位文本


def _is_within_workspace(path: Path) -> bool:
    """判断目标路径是否仍位于工作区内。"""

    return path == WORKSPACE_ROOT or WORKSPACE_ROOT in path.parents


def resolve_workspace_path(raw_path: str) -> Path:
    """把用户传入路径解析成绝对路径，并拦截越界访问。"""

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    if not _is_within_workspace(resolved):
        raise ValueError(f"path escapes workspace root: {WORKSPACE_ROOT}")
    return resolved


def parse_bool_env(value: str | None, default: bool) -> bool:
    """把环境变量中的布尔值解析成 True/False。"""

    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def parse_trace_console_mode(value: str | None, default: str = "summary") -> str:
    """解析轨迹输出模式，兼容旧的布尔配置写法。"""

    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return "summary"
    if normalized in {"0", "false", "no", "off"}:
        return "off"
    if normalized in {"summary", "full", "silent", "off"}:
        return normalized
    return default


def load_trace_settings() -> TraceSettings:
    """加载轨迹输出相关配置。"""

    # 允许通过旧变量 AGENT_TRACE 直接开关，也允许用 AGENT_TRACE_CONSOLE 细分模式。
    console_mode = parse_trace_console_mode(
        os.environ.get("AGENT_TRACE_CONSOLE") or os.environ.get("AGENT_TRACE"),
        default="summary",
    )
    html_enabled = parse_bool_env(
        os.environ.get("AGENT_TRACE_HTML"),
        default=console_mode != "off",
    )

    try:
        max_chars = int(os.environ.get("AGENT_TRACE_MAX_CHARS", str(TRACE_MAX_CHARS)))
    except ValueError:
        max_chars = TRACE_MAX_CHARS

    try:
        html_max_chars = int(
            os.environ.get("AGENT_TRACE_HTML_MAX_CHARS", str(TRACE_HTML_MAX_CHARS))
        )
    except ValueError:
        html_max_chars = TRACE_HTML_MAX_CHARS

    return TraceSettings(
        console_mode=console_mode,
        max_chars=max(200, max_chars),
        html_enabled=html_enabled,
        html_max_chars=max(500, html_max_chars),
    )


def truncate_text(text: str, max_chars: int) -> str:
    """限制单段文本长度，避免终端输出过长。"""

    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return f"{text[:max_chars]}\n... [truncated {hidden} chars]"


def prepare_trace_value(value: Any, max_chars: int) -> Any:
    """递归裁剪过长字段，方便把请求和响应结构打印到终端。"""

    if isinstance(value, str):
        return truncate_text(value, max_chars)
    if isinstance(value, list):
        return [prepare_trace_value(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: prepare_trace_value(item, max_chars) for key, item in value.items()}
    return value


def format_trace_data(data: Any, max_chars: int) -> str:
    """把结构化数据格式化为适合终端展示的 JSON。"""

    prepared = prepare_trace_value(data, max_chars)
    try:
        return json.dumps(prepared, ensure_ascii=False, indent=2)
    except TypeError:
        return truncate_text(str(prepared), max_chars)


def one_line_preview(value: Any, max_chars: int) -> str:
    """把结构化内容压缩为单行摘要，适合打印简短进度。"""

    text = format_trace_data(value, max_chars=max_chars)
    collapsed = " ".join(text.split())
    return truncate_text(collapsed, max_chars)


def print_trace_section(title: str, body: str) -> None:
    """在终端中打印一段可读性更强的轨迹区块。"""

    line = "=" * 24
    print(f"\n{line} {title} {line}")
    print(body)
    print("=" * (len(line) * 2 + len(title) + 2))


def format_subprocess_command(argv: list[str]) -> str:
    """把子进程参数列表格式化成便于人阅读的命令字符串。"""

    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def is_client_disconnect_error(exc: BaseException) -> bool:
    """判断异常是否属于浏览器主动断开连接这类可忽略情况。"""

    if isinstance(exc, CLIENT_DISCONNECT_ERRORS):
        return True

    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in CLIENT_DISCONNECT_WINERRORS:
            return True
        if exc.errno in CLIENT_DISCONNECT_ERRNOS:
            return True

    if isinstance(exc, ValueError) and "closed file" in str(exc).lower():
        return True

    return False


def build_trace_recorder(settings: TraceSettings) -> TraceRecorder:
    """创建当前会话的轨迹记录器。"""

    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_html_path = settings.trace_dir / "latest.html"
    session_html_path = settings.trace_dir / f"trace_{timestamp}.html"
    return TraceRecorder(
        settings=settings,
        session_label=timestamp,
        latest_html_path=latest_html_path,
        session_html_path=session_html_path,
    )


def start_live_reload_server(recorder: TraceRecorder) -> None:
    """启动一个本地 SSE 服务，只有 trace 发生变更时才通知 HTML 刷新。"""

    broadcaster = LiveReloadBroadcaster()

    class TraceHTTPServer(ThreadingHTTPServer):
        def handle_error(self, request: Any, client_address: Any) -> None:
            exc = sys.exc_info()[1]
            if exc is not None and is_client_disconnect_error(exc):
                return
            super().handle_error(request, client_address)

    class LiveReloadHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            try:
                if self.path in {"/", "/index.html", "/viewer", "/viewer/latest"}:
                    html_text = render_trace_html(
                        recorder=recorder,
                        live_reload_url_override="/events",
                        live_viewer_url_override=recorder.live_viewer_url,
                        served_via_http=True,
                    )
                    encoded = html_text.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return

                if self.path != "/events":
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                last_seen = broadcaster.version
                while True:
                    with broadcaster.condition:
                        broadcaster.condition.wait(timeout=30)
                        if broadcaster.version == last_seen:
                            payload = ": keep-alive\n\n"
                        else:
                            last_seen = broadcaster.version
                            data = json.dumps({"version": last_seen}, ensure_ascii=False)
                            payload = f"data: {data}\n\n"

                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
            except BaseException as exc:
                if is_client_disconnect_error(exc):
                    return
                raise

    server = TraceHTTPServer(("127.0.0.1", 0), LiveReloadHandler)
    host, port = server.server_address
    recorder.live_reload_broadcaster = broadcaster
    recorder.live_reload_server = server
    recorder.live_reload_url = f"http://{host}:{port}/events"
    recorder.live_viewer_url = f"http://{host}:{port}/viewer/latest"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def render_trace_html(
    recorder: TraceRecorder,
    live_reload_url_override: str | None = None,
    live_viewer_url_override: str | None = None,
    served_via_http: bool = False,
) -> str:
    """把轨迹事件渲染为带可折叠面板的 HTML 页面。"""

    event_blocks: list[str] = []
    total_events = len(recorder.events)

    for index, event in enumerate(recorder.events):
        event_class = escape(event["kind"])
        title = escape(event["title"])
        summary = escape(event["summary"])
        timestamp = escape(event["timestamp"])
        event_id = escape(f"{event['kind']}-{index}")
        payload = escape(
            format_trace_data(event["payload"], recorder.settings.html_max_chars)
        )
        open_attr = " open" if index == total_events - 1 else ""
        event_blocks.append(
            f"""
            <article class="event {event_class}">
              <details data-event-id="{event_id}"{open_attr}>
                <summary>
                  <span class="badge">{title}</span>
                  <span class="summary">{summary}</span>
                  <span class="timestamp">{timestamp}</span>
                </summary>
                <pre>{payload}</pre>
              </details>
            </article>
            """
        )

    event_html = "\n".join(event_blocks) or "<p class='empty'>暂无轨迹事件。</p>"
    generated_at = escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    workspace = escape(str(WORKSPACE_ROOT))
    session_label = escape(recorder.session_label)
    live_reload_url = json.dumps(
        live_reload_url_override if live_reload_url_override is not None else (recorder.live_reload_url or ""),
        ensure_ascii=False,
    )
    live_viewer_url = json.dumps(
        live_viewer_url_override if live_viewer_url_override is not None else (recorder.live_viewer_url or ""),
        ensure_ascii=False,
    )
    live_reload_text = (
        "Live reload on change"
        if (live_reload_url_override if live_reload_url_override is not None else recorder.live_reload_url)
        else "Live reload unavailable"
    )
    open_mode_text = (
        "Opened from local viewer"
        if served_via_http
        else "Open in browser to auto-switch to local viewer"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Trace Viewer</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --panel: #fffdf8;
      --border: #dccfbd;
      --text: #29231d;
      --muted: #736554;
      --accent: #b45f35;
      --accent-soft: #f0d7c7;
      --code-bg: #1f1d1a;
      --code-text: #f3eee7;
      --shadow: 0 18px 45px rgba(74, 52, 31, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, #f7d7c2 0, transparent 26%),
        linear-gradient(180deg, #fbf6ef 0%, var(--bg) 100%);
      color: var(--text);
      line-height: 1.5;
    }}
    .page {{
      width: min(1120px, calc(100vw - 32px));
      margin: 32px auto 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,248,240,0.95));
      border: 1px solid rgba(180, 95, 53, 0.18);
      border-radius: 24px;
      padding: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 3vw, 40px);
    }}
    .meta {{
      margin: 6px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .meta code {{
      color: var(--text);
      background: rgba(180, 95, 53, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .hint {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .refresh-note {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(180, 95, 53, 0.08);
      color: var(--accent);
      font-size: 13px;
      font-weight: 600;
    }}
    .events {{
      margin-top: 22px;
      display: grid;
      gap: 14px;
    }}
    .event {{
      background: rgba(255, 253, 248, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 28px rgba(64, 49, 31, 0.07);
      overflow: hidden;
    }}
    details {{
      width: 100%;
    }}
    summary {{
      list-style: none;
      cursor: pointer;
      padding: 18px 20px;
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
    }}
    summary::-webkit-details-marker {{
      display: none;
    }}
    .badge {{
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      flex: 0 0 auto;
    }}
    .summary {{
      font-weight: 600;
      flex: 1 1 360px;
    }}
    .timestamp {{
      color: var(--muted);
      font-size: 13px;
      flex: 0 0 auto;
    }}
    pre {{
      margin: 0;
      padding: 18px 20px 22px;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      background: var(--code-bg);
      color: var(--code-text);
      font-family: Consolas, "Cascadia Code", monospace;
      font-size: 13px;
      line-height: 1.55;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .empty {{
      background: var(--panel);
      border-radius: 18px;
      border: 1px dashed var(--border);
      padding: 20px;
      color: var(--muted);
    }}
    @media (max-width: 720px) {{
      .page {{
        width: min(100vw - 20px, 1120px);
        margin: 16px auto 28px;
      }}
      .hero {{
        padding: 20px;
      }}
      summary {{
        padding: 16px;
      }}
      pre {{
        padding: 16px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Agent Trace Viewer</h1>
      <p class="meta">Workspace: <code>{workspace}</code></p>
      <p class="meta">Session: <code>{session_label}</code></p>
      <p class="meta">Updated: <code>{generated_at}</code></p>
      <p class="hint">终端只显示摘要时，可以在这里展开查看完整请求、回复、工具参数和工具结果。</p>
      <p class="refresh-note">{live_reload_text}</p>
      <p class="refresh-note">{open_mode_text}</p>
    </section>
    <section class="events">
      {event_html}
    </section>
  </main>
  <script>
    const TRACE_STORAGE_KEY = "agent-trace-viewer-state";
    const liveReloadUrl = {live_reload_url};
    const liveViewerUrl = {live_viewer_url};

    function saveState() {{
      const openIds = Array.from(document.querySelectorAll("details[data-event-id]"))
        .filter((detail) => detail.open)
        .map((detail) => detail.dataset.eventId);
      const state = {{
        openIds,
        scrollY: window.scrollY,
      }};
      localStorage.setItem(TRACE_STORAGE_KEY, JSON.stringify(state));
    }}

    function restoreState() {{
      const raw = localStorage.getItem(TRACE_STORAGE_KEY);
      if (!raw) {{
        return;
      }}

      try {{
        const state = JSON.parse(raw);
        const openIds = new Set(state.openIds || []);
        document.querySelectorAll("details[data-event-id]").forEach((detail) => {{
          if (openIds.has(detail.dataset.eventId)) {{
            detail.open = true;
          }}
        }});
        if (typeof state.scrollY === "number") {{
          window.scrollTo({{ top: state.scrollY, behavior: "auto" }});
        }}
      }} catch (_error) {{
        // 状态恢复失败时忽略，保留页面可用性。
      }}
    }}

    window.addEventListener("beforeunload", saveState);
    window.addEventListener("pagehide", saveState);
    document.querySelectorAll("details[data-event-id]").forEach((detail) => {{
      detail.addEventListener("toggle", saveState);
    }});
    restoreState();

    if (window.location.protocol === "file:" && liveViewerUrl) {{
      saveState();
      window.location.replace(liveViewerUrl);
    }}

    if (liveReloadUrl) {{
      const eventSource = new EventSource(liveReloadUrl);
      eventSource.onmessage = () => {{
        saveState();
        window.location.reload();
      }};
    }}
  </script>
</body>
</html>
"""


def write_trace_html(recorder: TraceRecorder) -> None:
    """把当前轨迹同步写入 latest.html 和当前会话文件。"""

    if not recorder.settings.html_enabled:
        return

    try:
        html_text = render_trace_html(recorder)
        recorder.latest_html_path.write_text(html_text, encoding="utf-8")
        recorder.session_html_path.write_text(html_text, encoding="utf-8")
        if recorder.live_reload_broadcaster is not None:
            with recorder.live_reload_broadcaster.condition:
                recorder.live_reload_broadcaster.version += 1
                recorder.live_reload_broadcaster.condition.notify_all()
    except OSError:
        return


def emit_trace_event(
    recorder: TraceRecorder,
    kind: str,
    title: str,
    summary: str,
    payload: Any,
) -> None:
    """记录一次轨迹事件，并根据配置输出到终端或 HTML。"""

    event = {
        "kind": kind,
        "title": title,
        "summary": summary,
        "payload": payload,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    recorder.events.append(event)

    if recorder.settings.console_mode == "full":
        print_trace_section(title=title, body=format_trace_data(payload, recorder.settings.max_chars))
    elif recorder.settings.console_mode == "summary":
        print(summary)

    write_trace_html(recorder)


def describe_trace_mode(settings: TraceSettings) -> str:
    """把当前轨迹配置转成人类可读的简短描述。"""

    html_part = "html on" if settings.html_enabled else "html off"
    return f"{settings.console_mode} + {html_part}"


def load_dotenv_file(dotenv_path: Path | None = None) -> None:
    """从当前工作区的 .env 文件加载环境变量。"""

    path = dotenv_path or (WORKSPACE_ROOT / DOTENV_FILE_NAME)
    if not path.is_file():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # 兼容 `export KEY=value` 这种写法。
            if line.startswith("export "):
                line = line[7:].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            # 兼容简单的单双引号包裹值。
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            # 如果系统环境变量里已经有值，则不被 .env 覆盖。
            os.environ.setdefault(key, value)
    except OSError:
        # 读取 .env 失败时保持静默，后续仍按普通环境变量流程处理。
        return


def shell_exec(command: str) -> ToolExecutionResult:
    """执行 shell 命令，并返回 stdout/stderr 和实际执行细节。"""

    # Windows 下走 PowerShell，其他平台走 /bin/sh，保持行为尽量直观。
    if os.name == "nt":
        cmd = ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", command]
    else:
        cmd = ["/bin/sh", "-lc", command]

    execution = {
        "kind": "subprocess",
        "tool_name": "shell_exec",
        "requested_command": command,
        "argv": cmd,
        "display_command": format_subprocess_command(cmd),
        "cwd": str(WORKSPACE_ROOT),
        "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
    }

    try:
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        execution["returncode"] = result.returncode
        execution["status"] = "completed"
        return ToolExecutionResult(
            content=_normalize_output(result.stdout, result.stderr, result.returncode),
            execution=execution,
        )
    except subprocess.TimeoutExpired:
        execution["status"] = "timeout"
        return ToolExecutionResult(
            content=f"[error] command timed out after {COMMAND_TIMEOUT_SECONDS}s",
            execution=execution,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        execution["status"] = "error"
        execution["error"] = str(exc)
        return ToolExecutionResult(content=f"[error] {exc}", execution=execution)


def file_read(path: str) -> ToolExecutionResult:
    """读取工作区内的文本文件，并记录实际读取路径。"""

    execution = {
        "kind": "filesystem",
        "tool_name": "file_read",
        "operation": "read_text",
        "requested_path": path,
    }
    try:
        resolved = resolve_workspace_path(path)
        execution["resolved_path"] = str(resolved)
        execution["status"] = "completed"
        return ToolExecutionResult(
            content=resolved.read_text(encoding="utf-8"),
            execution=execution,
        )
    except Exception as exc:
        execution["status"] = "error"
        execution["error"] = str(exc)
        return ToolExecutionResult(content=f"[error] {exc}", execution=execution)


def file_write(path: str, content: str) -> ToolExecutionResult:
    """把文本内容写入工作区内文件，并记录实际写入路径。"""

    execution = {
        "kind": "filesystem",
        "tool_name": "file_write",
        "operation": "write_text",
        "requested_path": path,
        "content_length": len(content),
    }
    try:
        resolved = resolve_workspace_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        execution["resolved_path"] = str(resolved)
        execution["status"] = "completed"
        return ToolExecutionResult(
            content=f"OK - wrote {len(content)} chars to {resolved.relative_to(WORKSPACE_ROOT)}",
            execution=execution,
        )
    except Exception as exc:
        execution["status"] = "error"
        execution["error"] = str(exc)
        return ToolExecutionResult(content=f"[error] {exc}", execution=execution)


def python_exec(code: str) -> ToolExecutionResult:
    """在子进程中执行 Python 代码，并返回输出和实际命令。"""

    tmp_path: Path | None = None
    execution = {
        "kind": "subprocess",
        "tool_name": "python_exec",
        "operation": "python_exec",
        "cwd": str(WORKSPACE_ROOT),
        "python_executable": sys.executable,
        "timeout_seconds": PYTHON_TIMEOUT_SECONDS,
        "code_length": len(code),
    }
    try:
        # 先把代码落到临时文件，再用当前解释器启动子进程执行。
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)
            execution["temp_script_path"] = str(tmp_path)

        cmd = [sys.executable, str(tmp_path)]
        execution["argv"] = cmd
        execution["display_command"] = format_subprocess_command(cmd)

        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT_SECONDS,
        )
        execution["returncode"] = result.returncode
        execution["status"] = "completed"
        return ToolExecutionResult(
            content=_normalize_output(result.stdout, result.stderr, result.returncode),
            execution=execution,
        )
    except subprocess.TimeoutExpired:
        execution["status"] = "timeout"
        return ToolExecutionResult(
            content=f"[error] execution timed out after {PYTHON_TIMEOUT_SECONDS}s",
            execution=execution,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        execution["status"] = "error"
        execution["error"] = str(exc)
        return ToolExecutionResult(content=f"[error] {exc}", execution=execution)
    finally:
        # 无论执行成功与否，都尽量回收临时脚本文件。
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


# TOOLS 是 Agent 和模型之间的“工具总表”：
# 模型通过 schema 知道有哪些工具可用，代码通过 function 真正执行对应动作。
TOOLS: dict[str, ToolSpec] = {
    "shell_exec": ToolSpec(
        function=shell_exec,
        schema={
            "type": "function",
            "function": {
                "name": "shell_exec",
                "description": "Execute a shell command in the current workspace and return its output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
    ),
    "file_read": ToolSpec(
        function=file_read,
        schema={
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read a UTF-8 text file inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path to a file in the workspace.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
    ),
    "file_write": ToolSpec(
        function=file_write,
        schema={
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a UTF-8 text file inside the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path to a file in the workspace.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The UTF-8 text content to write.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ),
    "python_exec": ToolSpec(
        function=python_exec,
        schema={
            "type": "function",
            "function": {
                "name": "python_exec",
                "description": "Execute Python code in a subprocess and return stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python source code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
    ),
}


def load_openai_client() -> Any:
    """按环境变量配置创建 OpenAI 兼容客户端。"""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'openai'. Run: python -m pip install -r requirements.txt"
        ) from exc

    settings = load_client_settings()
    kwargs: dict[str, Any] = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAI(**kwargs), settings


def load_client_settings() -> ClientSettings:
    """按优先级读取 API Key、模型名和可选 base_url。"""

    # 先尝试从当前目录的 .env 中补充环境变量。
    load_dotenv_file()

    agent_api_key = os.environ.get("AGENT_API_KEY")
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # 优先使用通用 AGENT_* 配置，其次兼容文章中的 DeepSeek 方案，再兼容 OpenAI。
    api_key = agent_api_key or deepseek_api_key or openai_api_key
    if not api_key:
        raise RuntimeError(
            "No API key found. Set AGENT_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY."
        )

    base_url = os.environ.get("AGENT_BASE_URL")
    model = os.environ.get("AGENT_MODEL")

    # 如果当前使用的是 DeepSeek Key，就补上更贴近文章默认值的配置。
    if api_key == deepseek_api_key:
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        model = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    else:
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        model = model or os.environ.get("OPENAI_MODEL")

    if not model:
        raise RuntimeError(
            "No model found. Set AGENT_MODEL, DEEPSEEK_MODEL, or OPENAI_MODEL."
        )

    return ClientSettings(api_key=api_key, model=model, base_url=base_url)


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """把工具参数从 JSON 字符串解析成 Python 字典。"""

    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tool arguments: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to an object")
    return parsed


def extract_text_content(content: Any) -> str:
    """兼容不同响应格式，尽量提取出最终可展示文本。"""

    if isinstance(content, str):
        return content.strip() or "(empty response)"

    if isinstance(content, list):
        text_parts: list[str] = []
        # OpenAI 兼容接口有时会把内容拆成 content blocks，这里只提取文本块。
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        joined = "\n".join(part for part in text_parts if part)
        return joined.strip() or "(empty response)"

    return "(empty response)"


def execute_tool_call(tool_name: str, raw_arguments: str) -> ToolExecutionResult:
    """根据工具名分发调用，并返回工具结果及执行细节。"""

    tool = TOOLS.get(tool_name)
    if tool is None:
        return ToolExecutionResult(
            content=f"[error] unknown tool: {tool_name}",
            execution={
                "kind": "dispatcher",
                "tool_name": tool_name,
                "status": "error",
                "error": f"unknown tool: {tool_name}",
            },
        )

    try:
        arguments = parse_tool_arguments(raw_arguments)
        return tool.function(**arguments)
    except TypeError as exc:
        return ToolExecutionResult(
            content=f"[error] invalid arguments for {tool_name}: {exc}",
            execution={
                "kind": "dispatcher",
                "tool_name": tool_name,
                "status": "error",
                "raw_arguments": raw_arguments,
                "error": f"invalid arguments for {tool_name}: {exc}",
            },
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return ToolExecutionResult(
            content=f"[error] tool execution failed: {exc}",
            execution={
                "kind": "dispatcher",
                "tool_name": tool_name,
                "status": "error",
                "raw_arguments": raw_arguments,
                "error": f"tool execution failed: {exc}",
            },
        )


def visualize_request(
    turn: int,
    model: str,
    messages: list[dict[str, Any]],
    recorder: TraceRecorder,
) -> None:
    """记录当前轮次发给 LLM 的请求参数。"""

    payload = {
        "turn": turn,
        "model": model,
        "messages": messages,
        "tools": [tool.schema for tool in TOOLS.values()],
    }
    emit_trace_event(
        recorder=recorder,
        kind="llm-request",
        title=f"Turn {turn} | LLM Request",
        summary=f"[Turn {turn}] 发送 LLM 请求，messages={len(messages)}，tools={len(TOOLS)}",
        payload=payload,
    )


def visualize_response(turn: int, message: Any, recorder: TraceRecorder) -> None:
    """记录当前轮次模型返回的原始消息。"""

    tool_calls = message.tool_calls or []
    if tool_calls:
        tool_names = ", ".join(tool_call.function.name for tool_call in tool_calls)
        summary = f"[Turn {turn}] LLM 请求 {len(tool_calls)} 个工具: {tool_names}"
    else:
        final_text = extract_text_content(message.content)
        summary = f"[Turn {turn}] LLM 生成最终答复，长度 {len(final_text)} chars"

    emit_trace_event(
        recorder=recorder,
        kind="llm-response",
        title=f"Turn {turn} | LLM Response",
        summary=summary,
        payload=message.model_dump(exclude_none=True),
    )


def visualize_tool_call(turn: int, tool_call: Any, recorder: TraceRecorder) -> None:
    """记录模型请求调用的工具及参数。"""

    try:
        arguments: Any = parse_tool_arguments(tool_call.function.arguments)
    except ValueError:
        arguments = tool_call.function.arguments

    payload = {
        "turn": turn,
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.function.name,
        "arguments": arguments,
    }
    emit_trace_event(
        recorder=recorder,
        kind="tool-call",
        title=f"Turn {turn} | Tool Call",
        summary=(
            f"[Turn {turn}] 调用工具 {tool_call.function.name} "
            f"{one_line_preview(arguments, max_chars=120)}"
        ),
        payload=payload,
    )


def summarize_tool_execution(tool_name: str, execution: dict[str, Any]) -> str:
    """把工具实际执行细节压缩成一行摘要。"""

    kind = execution.get("kind")
    if kind == "subprocess":
        display_command = execution.get("display_command")
        if display_command:
            return f"实际执行命令: {truncate_text(display_command, 180)}"
    if kind == "filesystem":
        operation = execution.get("operation", "filesystem")
        resolved_path = execution.get("resolved_path") or execution.get("requested_path")
        return f"实际执行 {operation}: {truncate_text(str(resolved_path), 180)}"
    if kind == "dispatcher":
        error = execution.get("error", "dispatcher error")
        return f"调度失败: {truncate_text(str(error), 180)}"
    return f"执行细节: {one_line_preview(execution, max_chars=180)}"


def visualize_tool_execution(
    turn: int,
    tool_name: str,
    execution: dict[str, Any],
    recorder: TraceRecorder,
) -> None:
    """记录工具真正落到本机后的执行命令或文件操作。"""

    payload = {
        "turn": turn,
        "tool_name": tool_name,
        "execution": execution,
    }
    emit_trace_event(
        recorder=recorder,
        kind="tool-execution",
        title=f"Turn {turn} | Tool Execution",
        summary=f"[Turn {turn}] 工具 {tool_name} {summarize_tool_execution(tool_name, execution)}",
        payload=payload,
    )


def visualize_tool_result(turn: int, tool_name: str, result: str, recorder: TraceRecorder) -> None:
    """记录工具执行结果，帮助观察 Agent 的中间进度。"""

    status = "失败" if result.startswith("[error]") else "完成"
    preview = one_line_preview(result, max_chars=140)
    payload = {
        "turn": turn,
        "tool_name": tool_name,
        "result": result,
    }
    emit_trace_event(
        recorder=recorder,
        kind="tool-result",
        title=f"Turn {turn} | Tool Result",
        summary=f"[Turn {turn}] 工具 {tool_name} {status}，结果预览: {preview}",
        payload=payload,
    )


def agent_loop(
    user_message: str,
    messages: list[dict[str, Any]],
    client: Any,
    model: str,
    recorder: TraceRecorder,
) -> str:
    """运行文章中提到的极简 ReAct Agent Loop。"""

    # 用户输入先进入上下文，后续每一轮都在这份 messages 上追加观察结果。
    messages.append({"role": "user", "content": user_message})

    for _turn in range(1, MAX_TURNS + 1):
        # 1. 让模型基于当前上下文推理，并决定是否调用工具。
        visualize_request(turn=_turn, model=model, messages=messages, recorder=recorder)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool.schema for tool in TOOLS.values()],
        )
        message = response.choices[0].message
        visualize_response(turn=_turn, message=message, recorder=recorder)
        messages.append(message.model_dump(exclude_none=True))

        # 2. 如果模型不再请求工具，说明它准备直接给最终答案了。
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return extract_text_content(message.content)

        # 3. 逐个执行工具，并把工具观察结果写回上下文，供下一轮继续推理。
        for tool_call in tool_calls:
            visualize_tool_call(turn=_turn, tool_call=tool_call, recorder=recorder)
            tool_result = execute_tool_call(
                tool_name=tool_call.function.name,
                raw_arguments=tool_call.function.arguments,
            )
            visualize_tool_execution(
                turn=_turn,
                tool_name=tool_call.function.name,
                execution=tool_result.execution,
                recorder=recorder,
            )
            visualize_tool_result(
                turn=_turn,
                tool_name=tool_call.function.name,
                result=tool_result.content,
                recorder=recorder,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result.content,
                }
            )

    return f"[agent] reached maximum turns ({MAX_TURNS}), stopping."


def run_repl() -> int:
    """提供一个最小可用的命令行 REPL 交互入口。"""

    try:
        client, settings = load_openai_client()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    trace = load_trace_settings()
    recorder = build_trace_recorder(trace)
    start_live_reload_server(recorder)
    write_trace_html(recorder)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Agent ready in {WORKSPACE_ROOT}")
    print(f"Model: {settings.model}")
    print(
        f"Trace: {describe_trace_mode(trace)} "
        f"(console max chars: {trace.max_chars}, html max chars: {trace.html_max_chars})"
    )
    if trace.html_enabled:
        print(f"Trace viewer: {recorder.latest_html_path}")
        if recorder.live_reload_url:
            print(f"Live reload: {recorder.live_reload_url}")
        if recorder.live_viewer_url:
            print(f"Live viewer URL: {recorder.live_viewer_url}")
    print("Commands: exit | clear | trace summary | trace full | trace silent | trace off | trace path")
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered == "exit":
            print("Bye.")
            return 0
        if lowered == "clear":
            # clear 只清空会话历史，保留 system prompt。
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(context cleared)\n")
            continue
        if lowered == "trace on":
            trace.console_mode = "summary"
            trace.html_enabled = True
            write_trace_html(recorder)
            print(f"(trace -> {describe_trace_mode(trace)})\n")
            continue
        if lowered == "trace summary":
            trace.console_mode = "summary"
            trace.html_enabled = True
            write_trace_html(recorder)
            print(f"(trace -> {describe_trace_mode(trace)})\n")
            continue
        if lowered == "trace full":
            trace.console_mode = "full"
            trace.html_enabled = True
            write_trace_html(recorder)
            print(f"(trace -> {describe_trace_mode(trace)})\n")
            continue
        if lowered == "trace silent":
            trace.console_mode = "silent"
            trace.html_enabled = True
            write_trace_html(recorder)
            print(f"(trace -> {describe_trace_mode(trace)})\n")
            continue
        if lowered == "trace off":
            trace.console_mode = "off"
            trace.html_enabled = False
            print(f"(trace -> {describe_trace_mode(trace)})\n")
            continue
        if lowered == "trace path":
            print(f"(trace file: {recorder.latest_html_path})")
            if recorder.live_viewer_url:
                print(f"(live viewer url: {recorder.live_viewer_url})")
            print()
            continue

        reply = agent_loop(user_input, messages, client, settings.model, recorder)
        print(f"\nAgent> {reply}\n")


def main() -> int:
    """程序主入口。"""

    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
