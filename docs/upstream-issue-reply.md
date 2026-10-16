# 回复上游 issue 的草稿（可直接粘贴）

> 对应：`sitingge/workboddy-proxy` issue #1
> 「我用 Claude Code 接这个 workbuddy😵😵😵‍💫要做好多奇怪的转接」
>
> 结论：**不改上游也能搞定** —— 把工具调用在中间层「注入 + 还原」。
> 下面这段是回帖正文，可直接复制到 GitHub。

---

我同样用它来接一个**需要工具调用**的客户端（不是 Claude Code，但问题一模一样），
折腾了几天，结论是：**不用改这个项目，在中间加一层就能搞定**。把思路放这儿，给后面踩同一个坑的人省点时间。

## 症结

`pickMessages()` 只取 `messages`，所以客户端发在 **`body.tools`** 里的工具定义**整个被丢掉** ——
模型收不到任何工具描述，于是「不知道自己能做什么」，或者瞎编参数。
（不是 bug，README 的「已知限制」里写明了。）

## 思路：注入 → 还原

**① 注入（客户端 → 上游）**
把 `body.tools` 渲染成**紧凑文字**塞进 `system` 提示词（`system` 会被转成 `--append-system-prompt`，
所以能送到），末尾加一句约定：

```
【调用工具的方式】需要调用工具时，只输出一行 JSON，不要任何其它文字：
{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"工具名","arguments":"{\"参数\":\"值\"}"}}]}
```

渲染时把 `enum` 压成 `参数名(a/b/c)`、必填项标 `[必填]`、描述截断 —— 35 个工具从 ~17KB 压到 ~8.6KB。

**② 还原（上游 → 客户端）**
模型会把这段 JSON 当**普通文本**放在 `content` 里（它的 stream-json 协议没有 `tool_calls` 通道），
所以在中间层把它抠出来，组装成标准 OpenAI 响应：

- 非流式：`{"choices":[{"message":{"tool_calls":[...]},"finish_reason":"tool_calls"}]}`
- 流式：自己合成 SSE（`delta.tool_calls` + `[DONE]`）

我是拿 Python 网关做的（约 120 行），但逻辑跟语言无关。

## ⚠️ 三个坑（比思路本身值钱）

**坑 1：一定要在指令里明确否认内置工具。**
这个 CLI 自带 `Write` / `Read` / `Bash` / `future_task` 等工具，模型会**优先用自己训练过的原生格式**
（DeepSeek 的 DSML）去调它们，把我们的 JSON 约定晾在一边：

```
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="Write">
<｜｜DSML｜｜ parameter name="file_path" string="true">/home/x/memory/user_a.md</｜｜DSML｜｜ parameter>
```

**两重伤害**：用户在聊天里看到一堆乱码；而且它认为「你要的生图工具不在我的内置列表里」，
直接回「人家没有生图权限」。

⇒ 指令里写死「**你没有任何内置工具**（Write/Read/Bash 均不存在，不要输出任何 DSML 标记）」，
并且中间层要**同时会解析 DSML**、把未声明的调用丢弃、把标记从正文里清干净。
只做一半不行（我一开始就是只做了 JSON 那一半）。

**坑 2：有工具时只能「先收全、再合成流」。**
模型把调用当文本吐，没法流式透传着改写，所以**这一路请求会失去打字字效果**（正确性优先）。
不带工具的普通聊天仍然逐字流式。

**坑 3：`arguments` 是模型生成的字符串，执行前必须校验是不是合法 JSON** ——
模型完全可能给你单引号版本，一个语法错就能让下游工具崩。

## 想省事的

我把这套东西整理成了一个零依赖的单文件库（含 59 个测试，其中一个直接用这次抓到的原文做回归）：

**https://github.com/Roy1118/openai-tool-adapter**

`apply_tools_to_messages()` 注入、`extract_tool_calls()` 还原（自动识别 JSON/DSML 两种协议）、
`filter_tool_calls()` 过滤、`strip_dsml()` 清洗。拷一个文件走就能用。
README 里把上面几个坑都写详细了。

如果你的场景是 **Claude Code**，只需要在最后一步把 tool_calls 再转成 Anthropic 的
`tool_use` content block（库里有现成的 `to_anthropic_tool_use()`），
另外注意 Claude Code 对 `tool_use` 的 id 回传有要求，得原样带回去。
