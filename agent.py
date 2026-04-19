from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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


@dataclass(slots=True)
class ClientSettings:
    """LLM 客户端配置。"""

    api_key: str
    model: str
    base_url: str | None = None


@dataclass(slots=True)
class ToolSpec:
    """工具注册项，同时保存执行函数和 OpenAI tools schema。"""

    function: Callable[..., str]
    schema: dict[str, Any]


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


def shell_exec(command: str) -> str:
    """执行 shell 命令，并返回 stdout/stderr。"""

    # Windows 下走 PowerShell，其他平台走 /bin/sh，保持行为尽量直观。
    if os.name == "nt":
        cmd = ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", command]
    else:
        cmd = ["/bin/sh", "-lc", command]

    try:
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        return _normalize_output(result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {COMMAND_TIMEOUT_SECONDS}s"
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"[error] {exc}"


def file_read(path: str) -> str:
    """读取工作区内的文本文件。"""

    try:
        resolved = resolve_workspace_path(path)
        return resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[error] {exc}"


def file_write(path: str, content: str) -> str:
    """把文本内容写入工作区内文件。"""

    try:
        resolved = resolve_workspace_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"OK - wrote {len(content)} chars to {resolved.relative_to(WORKSPACE_ROOT)}"
    except Exception as exc:
        return f"[error] {exc}"


def python_exec(code: str) -> str:
    """在子进程中执行 Python 代码，并返回输出。"""

    tmp_path: Path | None = None
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

        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT_SECONDS,
        )
        return _normalize_output(result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return f"[error] execution timed out after {PYTHON_TIMEOUT_SECONDS}s"
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"[error] {exc}"
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


def execute_tool_call(tool_name: str, raw_arguments: str) -> str:
    """根据工具名分发调用，并处理参数解析错误。"""

    tool = TOOLS.get(tool_name)
    if tool is None:
        return f"[error] unknown tool: {tool_name}"

    try:
        arguments = parse_tool_arguments(raw_arguments)
        return tool.function(**arguments)
    except TypeError as exc:
        return f"[error] invalid arguments for {tool_name}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"[error] tool execution failed: {exc}"


def agent_loop(user_message: str, messages: list[dict[str, Any]], client: Any, model: str) -> str:
    """运行文章中提到的极简 ReAct Agent Loop。"""

    # 用户输入先进入上下文，后续每一轮都在这份 messages 上追加观察结果。
    messages.append({"role": "user", "content": user_message})
    tool_schemas = [tool.schema for tool in TOOLS.values()]

    for _turn in range(1, MAX_TURNS + 1):
        # 1. 让模型基于当前上下文推理，并决定是否调用工具。
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        # 2. 如果模型不再请求工具，说明它准备直接给最终答案了。
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return extract_text_content(message.content)

        # 3. 逐个执行工具，并把工具观察结果写回上下文，供下一轮继续推理。
        for tool_call in tool_calls:
            result = execute_tool_call(
                tool_name=tool_call.function.name,
                raw_arguments=tool_call.function.arguments,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
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

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Agent ready in {WORKSPACE_ROOT}")
    print(f"Model: {settings.model}")
    print("Commands: exit | clear")
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

        reply = agent_loop(user_input, messages, client, settings.model)
        print(f"\nAgent> {reply}\n")


def main() -> int:
    """程序主入口。"""

    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
