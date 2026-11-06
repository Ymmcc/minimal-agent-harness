# Minimal Agent · Harness 笔试

一个最小可用 Agent：自行实现循环、工具注册、模型响应解析、SQLite 会话状态和基础上下文压缩。**不依赖任何 Agent 框架。**

提供命令行交互。两个终端即两个窗口；默认生成不同 session，也可指定 session 恢复。核心代码约 570 行（含注释、空行、离线演示模型）。

## 运行

Python 3.11+，本地验证环境 Python 3.13。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -X utf8 -m agent
```

macOS/Linux 使用 `.venv/bin/python`。以下命令中的 `python` 也可替换为虚拟环境的 Python 路径。

默认是**离线 mock**，无需 API Key。其规则仅用于确定性演示，不具备通用语言理解能力：

```text
我叫小林
我叫什么
计算 12*(3+4)
再乘以 2
记待办 周五复习 Agent
待办列表
完成待办 1
搜索 Agent runtime
/trace
/quit
```

`完成待办` 后的数字以查询结果中的真实 ID 为准，ID 在数据库中全局递增。

### 演示两个窗口和恢复

终端 1：

```powershell
python -X utf8 -m agent --user A --session weather
```

终端 2：

```powershell
python -X utf8 -m agent --user A --session report
```

分别新增待办后查询，状态互不影响。关闭再使用相同参数启动即可恢复。交互中 `/session weather` 可切换；不填 `--session` 每次启动生成新 ID 并显示。

### 连接真实模型

需使用支持 **Chat Completions 原生 function calling** 的模型，并支持本项目 JSON Schema（含 `allOf`/`if`/`then`）。此时工具选择完全由模型根据描述与 Schema 决定，Runtime 没有关键词路由。供应商支持的 Schema 子集可能不同，接入时应实际验证。

只在自己的终端设置环境变量。下面通过安全输入读取密钥，不将密钥写入命令历史、仓库或提交材料：

```powershell
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host '输入自己的模型密钥' -AsSecureString)).Password
$env:LLM_BASE_URL = Read-Host '输入供应商 Chat Completions 基础地址（到 /v1 或兼容前缀为止）'
$env:LLM_MODEL = Read-Host '输入支持工具调用的模型 ID'
python -X utf8 -m agent --live --user A --session interview
Remove-Item Env:LLM_API_KEY
```

程序在基础地址后追加 `/chat/completions`。远程地址要求 HTTPS；本地服务可用 localhost HTTP。禁止重定向，避免凭证被转发。密钥、`.env` 和本地数据库不应提交到仓库。

真实模式示例：先问“计算 18×7，再把结果记为复习待办”，然后问“把刚才的待办标为完成”。模型应先算、再写入、再根据 ID 更新；具体表现依赖所选模型。**本次未使用真实付费模型验证，这些自然语言示例是验收步骤，不是已完成的实测结论。**

## 设计

```text
CLI → Runtime.chat → 获取 session 租约 → 恢复中断记录
                         ↓
              ContextBuilder 构造上下文
                         ↓
              Model.complete → 解析规范化输出
                         ↓
              有工具调用？ ── 否 → 保存最终答案、返回
                  │ 是
                  ↓
          Schema 校验 → 执行工具 → 保存工具结果
                  └────────────── 回到模型（最多 6 步）
```

| 文件 | 职责 |
| --- | --- |
| `agent/__main__.py` | 输入、会话切换、输出、trace 查询 |
| `agent/runtime.py` | 自行实现的 Agent 主循环 |
| `agent/llm.py` | HTTP 适配、结构化响应解析、独立的离线 mock |
| `agent/tools.py` | 工具注册、JSON Schema 校验、三个工具 |
| `agent/store.py` | SQLite 消息、会话、待办、trace、租约 |
| `agent/context.py` | 上下文组装与完整旧轮次的摘录压缩 |

唯一直接第三方依赖 `jsonschema` 只做参数校验，不承担 Agent 编排。

### 模型输出解析与工具

- 有 `tool_calls` 时提取 `id`、名称、JSON 参数；没有时将 `content` 当最终答案。
- `content` 可包含面向用户的简短行动说明，随 assistant 消息保存。这是可观察的决策说明，不要求模型输出私密思维链；供应商的 `reasoning_content` 不存储、不回传。
- 空响应、截断、异常协议：停止本轮并保存可读错误；非法工具 JSON、参数不匹配、未知工具：作为工具失败结果回填，模型可在下一步修正。
- 一步最多 4 个工具调用，按顺序执行并按调用 ID 配对；最多 6 步（可在 Runtime 构造时改为 1–8）。
- `calculator`：AST 白名单，支持 `+ - * / %`、括号、一元正负；禁止 `eval`、函数调用、幂、访问属性；限制长度、节点数和数值范围。
- `search`：明确标记的本地 mock，不冒充联网搜索或实时天气。
- `todo`：新增、查询、完成；状态由 `(user, session)` 限定，用户信息由 Runtime 注入，模型参数不能越权切换身份。

### Session 与崩溃处理

所有会话数据按 `(user, sid)` 隔离，恢复时加载相同键的消息。SQLite 的短事务只覆盖本地状态操作，网络请求期间不持有数据库写锁。

`BEGIN IMMEDIATE` + 条件更新原子抢占 300 秒租约。同 session 的并发请求返回 busy，不静默丢弃或合并；不同 session 可继续。每次写消息、写摘要和改待办前检查 owner 与有效期，旧进程不能覆盖接管后的会话。

正常退出通过 `finally` 释放租约；进程被强杀后，最多等待 300 秒才能接管。恢复时为缺失的工具结果补 `interrupted`，不自动重放写操作。todo 新增以 `(user, sid, call_id)` 去重；同一业务请求若被模型换新 ID 再发，仍可能重复。因此不宣称外部副作用 exactly-once。CLI 是本地演示，`--user` 是命名空间，不是登录认证。

### Memory 召回时机与放置方式

本实现没有向量数据库或跨 session 用户画像，memory 指**当前会话的持久消息、历史摘录和待办业务状态**。

| 信息 | 何时读 | 放到哪里 |
| --- | --- | --- |
| 最近完整消息 / 当前轮调用链 | 每次模型调用前 | system 规则之后，保留原角色、顺序和调用配对 |
| 历史摘录 | 每次模型调用前；压缩后保存 | system 后的历史参考消息，明确为数据而非系统指令 |
| 当前待办 | 模型判断需要新增/查询/完成时 | `todo` 的工具结果；不凭旧摘要认定业务状态 |
| 完整原始历史 | SQLite 永久保留，供调试核对 | 不全部塞入模型 |

预算默认 18,000 **字符**，计入 system、历史和工具 Schema；这是近似预算，不是精确 token 数。超过阈值时，按完整旧轮次移出上下文，摘录用户目标、回答及工具结果；摘要约 2,000 字符，过长保留首尾。用 `summary_until` 记录已压缩消息 ID，防止重复摘录。保留当前轮；当前轮仍放不下则明确要求拆分。

这是有损压缩：可能遗漏很早的事实，不能保证 200 轮后任意细节都能召回。近期追问、待办持久状态和工具消息结构有测试保障；架构题说明了更完整的改进方案。单条工具结果超过 2,500 字符时包装成有效 JSON 摘要，不将截断文本冒充完整结果。

### 异常和日志

HTTP 单次 socket 超时 20 秒，不重试有副作用的工具。它不是严格的端到端 deadline。当前三个工具都是受限本地操作，没有通用线程超时器；接入网络或任意脚本工具时需要独立超时与进程隔离。

trace 存入本地 SQLite，包含 `run_id`、step、调用 ID、工具名称、成败和耗时。`/trace` 查询当前 session；工具结果和公开行动说明在消息表中。不会记录请求头、模型密钥或私密推理；HTTP 错误不打印供应商响应正文。运行数据含聊天内容，因此 `data/` 不提交。

## 测试与材料

```powershell
python -X utf8 -m unittest discover -v
```

测试通过脚本模型、真实 SQLite 和本地 HTTP 服务验证循环与适配器，不调用外部 LLM，不消耗额度。详细结果见 [测试报告](docs/TEST_REPORT.md)。

- [架构设计回答：五个模块各选一道](docs/ARCHITECTURE.md)
- [AI Prompt 与问题解决记录](docs/AI_USAGE.md)
