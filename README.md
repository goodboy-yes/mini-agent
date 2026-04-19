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

## Notes

- Windows 下 `shell_exec` 使用 `PowerShell`
- `python_exec` 会在子进程里运行 Python，并把当前工作目录设为 `test-one`
- 这是一个本地高权限 Agent 雏形，`shell_exec` 能执行命令，使用前要明确自己给了它什么权限
