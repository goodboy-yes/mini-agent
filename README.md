# Minimal Agent In `test-one`

这个项目参考了腾讯新闻文章《详尽地带你从零开始设计实现一个AI Agent框架》中的“二、AI Agent 框架实践篇”，按文章里的核心结构实现了一个可运行的极简 Agent：

- `Agent Loop`：`LLM call -> parse tool_calls -> execute -> append results -> loop/exit`
- `Tools Registry`：注册四个工具 `shell_exec / file_read / file_write / python_exec`
- `Context Engineering`：用 `messages` 列表保存 system、user、assistant、tool 消息
- `CLI REPL`：支持持续对话、`clear` 清上下文、`exit` 退出

和文章里的极简版相比，这个实现多补了两点实用能力：

- `file_read` 和 `file_write` 默认限制在当前工作区，避免把相对路径写到别处
- 模型配置改成环境变量驱动，既能直接用 DeepSeek，也能接 OpenAI 兼容接口
- 启动时会自动尝试读取当前目录下的 `.env` 文件
- 终端默认只显示精简进度摘要，完整轨迹会同步写入可折叠的 HTML 查看器

## Files

- [agent.py](./agent.py)：Agent 主程序
- [requirements.txt](./requirements.txt)：依赖
- [.env.example](./.env.example)：环境变量示例

## Install

```powershell
python -m pip install -r requirements.txt
```

## Configure

程序会优先读取当前目录下的 `.env` 文件；如果系统环境变量里已经存在同名变量，则系统环境变量优先。

```env
AGENT_API_KEY=your-key
AGENT_BASE_URL=https://your-openai-compatible-endpoint
AGENT_MODEL=your-model-name
AGENT_TRACE_CONSOLE=summary
AGENT_TRACE_HTML=1
AGENT_TRACE_MAX_CHARS=4000
AGENT_TRACE_HTML_MAX_CHARS=20000
```

### 方案 1：直接按文章里的 DeepSeek 方式运行

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxxxx"
python agent.py
```

可选变量：

```powershell
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-chat"
```

### 方案 2：使用任意 OpenAI 兼容接口

```powershell
$env:AGENT_API_KEY = "your-key"
$env:AGENT_BASE_URL = "https://your-openai-compatible-endpoint"
$env:AGENT_MODEL = "your-model-name"
python agent.py
```

### 方案 3：使用 OpenAI 官方接口

```powershell
$env:OPENAI_API_KEY = "sk-xxxxx"
$env:OPENAI_MODEL = "your-model-name"
python agent.py
```

如果你用的是 OpenAI 官方接口，一般不需要额外设置 `OPENAI_BASE_URL`。

## Usage

启动后可以直接输入自然语言任务，例如：

- `帮我列出当前目录有哪些文件`
- `创建一个 hello.py，内容是打印 hello world`
- `帮我统计当前目录下所有 .py 文件的代码行数`

内置命令：

- `clear`：清空当前会话上下文，但保留 system prompt
- `exit`：退出 REPL
- `trace summary`：终端显示精简摘要，同时保留 HTML 轨迹
- `trace full`：终端显示完整区块日志，同时保留 HTML 轨迹
- `trace silent`：终端不打印轨迹，只写入 HTML 轨迹
- `trace off`：关闭执行轨迹可视化
- `trace path`：显示当前 HTML 轨迹文件路径

程序启动后会生成 `.agent-traces/latest.html`，你可以在浏览器里打开它。每个事件都是可折叠面板，适合查看长请求、长回复和工具结果。
为了让热更新更稳定，Agent 运行时还会启动一个本地 HTTP 查看器，并在控制台打印 `Live viewer URL`。
现在页面只会在 trace 真正发生变更时刷新，不再按固定时间整页刷新，并尽量保留你的展开状态和滚动位置。
如果你直接用浏览器打开 `latest.html`，页面会自动跳转到本地查看器；如果你是在 IDE 的源码标签页里看文件内容，那依然不会自动刷新，因为那不是实际执行 HTML/JS 的页面。

HTML 轨迹里会记录以下区块：

- `LLM Request`：当前轮次发给模型的请求参数
- `LLM Response`：模型返回的原始消息
- `Tool Call`：模型请求调用的工具与参数
- `Tool Execution`：工具真正映射到本机后的执行命令或文件操作
- `Tool Result`：工具实际执行结果

如果你想调节终端和 HTML 的输出长度，可以通过 `.env` 或系统环境变量配置：

```env
AGENT_TRACE_CONSOLE=summary
AGENT_TRACE_HTML=1
AGENT_TRACE_MAX_CHARS=2000
AGENT_TRACE_HTML_MAX_CHARS=12000
```

如果你是在 IDE 里直接打开 HTML 源码文件标签页，那通常不会自动刷新，因为那是文本编辑器视图，不是浏览器/预览器视图。要看到热更新，请用浏览器打开 `latest.html`，或使用 IDE 的 HTML Preview。

## Notes

- Windows 下 `shell_exec` 使用 `PowerShell`
- `python_exec` 会在子进程里运行 Python，并把当前工作目录设为 `test-one`
- 这是一个本地高权限 Agent 雏形，`shell_exec` 能执行命令，使用前要明确自己给了它什么权限
