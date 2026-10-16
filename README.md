# openai-tool-adapter

**给「不支持 function calling」的 LLM 上游补上 OpenAI 工具调用能力** —— 零依赖，单个文件，拷走就能用。

起因很具体：我在一个中转网关里接了一个**不转发 `tools` 字段**的上游（它只读 `messages`，把
`tools` 整个丢掉）。结果是模型**知道自己能生图，却不知道参数怎么填** —— 要么不调用，要么瞎编参数。
上游作者在 README 里明确写了「不做工具调用」，唯一的 issue 也是这个问题，没人解决。

于是我在中间层把这件事补掉。后来又把踩到的坑都固化成了测试，就有了这个库。

---

## 它做什么

两个方向、三步处理：

```
客户端请求（带 tools）
      │
      │  ① 注入：把 tools 渲染成紧凑文字，塞进 system 提示词
      ▼
   不支持 tools 的上游  ──►  模型把「工具调用」当纯文本吐回来
      │
      │  ② 还原：从文本里抠出调用（认两种协议）
      │  ③ 过滤：丢掉客户端没声明过的工具（上游自带的那堆）
      ▼
标准 OpenAI 响应 / SSE（客户端直接可用）
```

| 函数 | 作用 |
|---|---|
| `render_tools(tools)` | 把 OpenAI tools 定义压成紧凑文字（比原样 JSON 省一半以上） |
| `apply_tools_to_messages(msgs, tools)` | ① **注入**：追加到 system，返回新列表（不改原对象） |
| `extract_tool_calls(text)` | ② **还原**：自动识别「JSON 约定」或「上游原生格式」 |
| `filter_tool_calls(calls, allowed)` | ③ **过滤**：丢掉客户端没声明的调用 |
| `strip_dsml(text)` | 把原生工具标记从正文里清干净（用户不会看到乱码） |
| `validate_tool_calls(calls)` | 执行前校验 `arguments` 是不是合法 JSON |
| `openai_tool_response(...)` | 组装非流式响应（`finish_reason: tool_calls`） |
| `sse_from_message(...)` | 组装流式 SSE（`delta.tool_calls` + `[DONE]`） |
| `to_anthropic_tool_use(calls)` | 顺手转成 Claude Messages API 的 `tool_use` block |

---

## 快速开始

```python
from adapter import (apply_tools_to_messages, extract_tool_calls,
                     filter_tool_calls, strip_dsml, validate_tool_calls,
                     openai_tool_response)

tool_names = [t["function"]["name"] for t in tools]

# ① 注入后发给上游
sent = apply_tools_to_messages(messages, tools)
reply = upstream.chat(sent)              # ← 上游把工具调用当纯文本吐回来

# ② 还原 + ③ 过滤
calls = extract_tool_calls(reply) or []
calls, dropped = filter_tool_calls(calls, tool_names)
if dropped:
    log.warning("丢弃上游自带工具调用: %s", [c["function"]["name"] for c in dropped])

if calls:
    ok, errors = validate_tool_calls(calls)
    if not ok:
        return {"error": errors}
    return openai_tool_response(model, calls)

# 普通聊天：别忘了把标记清干净，否则用户会看到一堆乱码
return {"content": strip_dsml(reply)}
```

完整可跑的中间层示例（含假上游，能直接观察注入与还原）：[`examples/proxy_shim.py`](examples/proxy_shim.py)

```bash
python examples/proxy_shim.py            # 起一个假上游 + 自测
python -m unittest discover -s tests -v  # 59 个测试
```

---

## 为什么需要它（四个真实踩过的坑）

这东西的价值全在坑里。每一条都固化成了测试。

### 坑 1：`tools` 可能在请求体顶层，而中转层只读 `messages`

上游「只取对话消息、忽略其它字段」是常见设计。**优化请求体时必须逐个字段过一遍** ——
只盯着 `messages` 裁剪，会「顺手优化掉整个功能」。

### 坑 2：往 system 里注入要**明确否认内置工具**，否则模型会用原生协议

这是我实际踩的最大的坑。上游 CLI 自带 `Write` / `Read` / `Bash` / `future_task` 等工具，
模型会用**它自己训练过的原生工具标记**（DeepSeek 叫 DSML）来"调用"，把我们的 JSON 约定晾在一边：

```
既然你当面问了，我顺手把你的名字记小本本了。
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="Write">
<｜｜DSML｜｜ parameter name="file_path" string="true">/home/x/memory/user_a.md</｜｜DSML｜｜ parameter>
...
```

**两重伤害**：用户在聊天里看到一堆乱码；同时它认为「生图工具不在我的内置列表里」，
于是直接回**「人家没有生图权限啦」**。

⇒ 注入指令里必须写死「**你没有任何内置工具**（Write/Read/Bash…均不存在）」，并且网关要
**同时认 DSML 格式**。只做一半都不行。

### 坑 3：还原出来的 `arguments` 是模型生成的**字符串**，必须校验

```python
# 模型完全可能给你这个：
{"name": "search", "arguments": "{'query': 'x'}"}   # 单引号 —— 不是合法 JSON
```
执行前不过一道 `validate_tool_calls()`，一个语法错就能让下游工具直接崩。

### 坑 4：展示给用户的 prompt 里，JSON 示例必须是**真 JSON**

```python
# ❌ 常见写法：本意是转义，结果模型看到的示例是坏 JSON
'{{"tool_calls":[{{"id":"call_1", ...}}]}}'
# ✅
'{"tool_calls":[{"id":"call_1", ...}]}'
```
功能表面能用（模型很聪明），但小模型容易照抄错。`test_instruction_example_is_valid_json`
就是专门盯这个的回归测试。

---

## 局限（说清楚，别让人踩同样的坑）

- 🔴 **有工具时得「先收全、再合成流」** —— 模型把调用当**文本**吐，流式透传没法回头改写进
  `delta.tool_calls`。所以**这一路请求会失去打字机效果**（正确性优先）。
  普通聊天（不带工具）仍然逐字流式。
- ⚠️ **依赖模型愿意遵守约定**。这是"提示词级"适配，不是协议级。指令越明确越稳，
  但仍然可能存在模型不听话的情况 —— 所以**过滤 + 校验**不能省。
- ⚠️ **注入会让 system 变长**（本例 35 个工具约 +8.6k token）。工具特别多时建议按需裁剪。

## 什么时候**不要**用它

如果上游支持原生 `tools` 字段（OpenAI 官方、以及绝大多数商业 API），**直接用原生能力**。
这个库存在的唯一理由是：上游不支持、且你改不动它。

---

## 测试

```bash
python -m unittest discover -s tests -v
```

`tests/test_adapter.py`（35 个）覆盖渲染/注入/还原/校验/响应组装；
`tests/test_dsml.py`（24 个）专门覆盖 DSML 解析、清洗与过滤，其中
`RealIncidentTest` 直接用线上抓到的原文做回归。

零第三方依赖，只用标准库 —— 拷进任何项目都不会引入依赖冲突。

## License

MIT
