#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adapter.py —— 给「不支持 function calling」的 LLM 上游补上 OpenAI 工具调用能力。

很多上游会**静默忽略**请求体里的 `tools` 字段：你把 35 个工具定义发过去，它当没看见。
表现是模型「知道自己能做某件事」，却**不知道参数该怎么填** —— 要么不调用，要么瞎编参数。

本库不指望上游配合，直接在中间层把这件事补掉，只有两步：

    ① 注入 inject —— 把 tools 定义渲染成紧凑文字塞进 **system 提示词**，
       并约定「需要调用工具时，只输出一行 tool_calls JSON」。

    ② 还原 parse back —— 从模型返回的**纯文本**里抠出那段 JSON，
       转成标准 OpenAI `tool_calls`（非流式给 JSON，流式给合成 SSE）。

**零第三方依赖**（只用标准库），可直接拷进任何项目。

典型用法：

    from adapter import (apply_tools_to_messages, extract_tool_calls,
                         openai_tool_response)

    messages = apply_tools_to_messages(messages, tools)   # ① 注入
    ... 把 messages 发给上游，拿到纯文本 reply ...
    calls = extract_tool_calls(reply)                      # ② 还原
    if calls:
        return openai_tool_response(model, calls, usage)

端到端可跑的中间层示例见 `examples/proxy_shim.py`。
"""

import json
import os
import re
import time

__all__ = [
    "TOOL_INSTRUCTION",
    "render_tools",
    "apply_tools_to_messages",
    "extract_tool_calls",
    "extract_dsml_tool_calls",
    "filter_tool_calls",
    "has_dsml",
    "strip_dsml",
    "validate_tool_calls",
    "openai_tool_response",
    "sse_from_message",
    "to_anthropic_tool_use",
]


# ⚠️ 两个坑都在这里：
#    ① 用 `str.replace("{tools}", ...)` 而不是 `str.format()` —— 正文里有大量花括号，format 会把它们当占位符。
#    ② 花括号必须是**单层**！写成 `{{"tool_calls"...}}` 的话，模型看到的示例就是**非法 JSON**。
#       （真实项目里就是这么错的：功能能用，但示例是坏的，小模型容易照抄错。）
#
# 🔴 必须明写「你没有内置工具」：如果上游 CLI/Agent 自带 Bash/Write/Read 之类的工具，
#    模型会优先用它**原生的**工具协议（如 DeepSeek 的 DSML），把我们的 JSON 约定晾在一边 ——
#    表现就是「明明给了生图工具，它却说没有生图权限」，并且把原生标记当正文吐出来。
TOOL_INSTRUCTION = """【可用工具】你可以在需要时调用下列工具（OpenAI function calling）：
{tools}

🔴【重要】你**没有任何内置工具**（Write / Read / Bash / Edit / future_task 等均不存在，不要使用、
不要输出任何形如 `<..DSML..>` 的标记）。上面列出的就是你**唯一**可用的工具。

【调用工具的方式】需要调用工具时，**只输出一行 JSON，不要任何其它文字、不要 Markdown 代码块**：
{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"工具名","arguments":"{\\"参数\\":\\"值\\"}"}}]}
一次可以调用多个（放在同一个数组里）。不需要工具时正常聊天回复即可。"""


def render_tools(tools):
    """把 OpenAI tools 定义压成**紧凑文字**（比原样 JSON 省一半以上字符）。

    兼容两种写法：
        {"type": "function", "function": {...}}   ← OpenAI 标准
        {"name": ..., "description": ..., "parameters": {...}}   ← 裸 function 对象

    enum 会写成 `参数名(a/b/c)`，必填项打 `[必填]`，描述截断到 80 字符。
    任何异常都返回 `""`（调用方无需 try）。
    """
    try:
        if not isinstance(tools, list) or not tools:
            return ""
        out = []
        for t in tools:
            fn = (t or {}).get("function") if isinstance(t, dict) else None
            fn = fn or (t if isinstance(t, dict) else {}) or {}
            name = fn.get("name")
            if not name:
                continue
            desc = re.sub(r"\s+", " ", str(fn.get("description") or "")).strip()
            params = fn.get("parameters") or {}
            props = (params.get("properties") or {}) if isinstance(params, dict) else {}
            required = set(params.get("required") or []) if isinstance(params, dict) else set()
            segs = []
            for key, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                d = re.sub(r"\s+", " ", str(spec.get("description") or "")).strip()
                s = key
                enum = spec.get("enum")
                if isinstance(enum, list) and enum:
                    s += "(" + "/".join(str(x) for x in enum[:10]) + ")"
                if d:
                    s += ":" + d[:80]
                if key in required:
                    s += "[必填]"
                segs.append(s)
            out.append("- %s: %s\n    参数: %s" % (name, desc[:240], "; ".join(segs) or "(无)"))
        return "\n".join(out)
    except Exception:
        return ""


def apply_tools_to_messages(messages, tools, instruction=TOOL_INSTRUCTION):
    """① 注入：把工具说明追加到 **system 消息** 末尾，返回**新的** messages。

    · 已有 system → 追加到它的 content 尾部
    · 没有 system → 在最前面插一条
    · tools 渲染为空 → 只做一份浅拷贝返回（调用方拿到的永远是新对象，原对象不会被改）

    ⚠️ 为什么必须放 system、而不是塞进 user：
       多数上游/CLI 只取「第一个 system + 最后一条 user」作为上下文，
       塞进 system 才能保证它**每一轮都在**。
    """
    msgs = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]
    txt = render_tools(tools)
    if not txt:
        return msgs
    block = instruction.replace("{tools}", txt)
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            m["content"] = content + "\n\n" + block
            return msgs
    msgs.insert(0, {"role": "system", "content": block})
    return msgs


def _first_json_object(text):
    """取出文本里第一个**括号配对**的 JSON 对象子串（能正确跳过字符串内部的花括号）。

    找不到返回 None。用来容忍「模型在 JSON 前后加了一句客套话」。
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def extract_tool_calls(text):
    """② 还原：从模型返回的纯文本里抠出 tool_calls，转成标准 OpenAI 结构。

    依次尝试两种协议：
      1. **我们注入的约定** —— `{"tool_calls": [...]}` JSON（容忍前后客套话、```json 代码块）
      2. **上游原生工具格式** —— 如 DeepSeek 的 DSML（见 `extract_dsml_tool_calls`）

    抠不出来或结构不对 → 返回 **None**（调用方据此判断「这就是一次普通聊天」）。
    """
    return _extract_json_tool_calls(text) or extract_dsml_tool_calls(text)


def _extract_json_tool_calls(text):
    """只认 `{"tool_calls": [...]}` JSON 写法（内部用）。"""
    try:
        s = (text or "").strip()
        if not s or "tool_calls" not in s:
            return None
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

        data = None
        if s.startswith("{"):
            try:
                data = json.loads(s)
            except Exception:
                data = None
        if data is None:                       # 模型可能在 JSON 前后加了话
            obj = _first_json_object(s)
            if not obj:
                return None
            data = json.loads(obj)

        calls = data.get("tool_calls") if isinstance(data, dict) else None
        if not isinstance(calls, list) or not calls:
            return None

        out = []
        for i, c in enumerate(calls):
            if not isinstance(c, dict):
                return None
            fn = c.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if not name:
                return None
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps({} if args is None else args, ensure_ascii=False)
            out.append({"id": c.get("id") or ("call_%d" % (i + 1)),
                        "type": "function",
                        "function": {"name": name, "arguments": args}})
        return out
    except Exception:
        return None


# ─────────────── 上游原生工具格式：DeepSeek DSML ───────────────
# 模型有时不用我们的 JSON 约定，而是吐它"原生训练"的工具协议，例如：
#
#   <｜｜DSML｜｜tool_calls>
#   <｜｜DSML｜｜invoke name="Write">
#   <｜｜DSML｜｜parameter name="file_path" string="true">/tmp/a.txt</｜｜DSML｜｜parameter>
#   </｜｜DSML｜｜invoke>
#   </｜｜DSML｜｜tool_calls>
#
# 这些标记会被客户端**当成正文显示**（用户看到一堆乱码），必须还原成 tool_calls。
#
# ⚠️ 分隔符是全角竖线 `｜`(U+FF5C)，但有些字体/转发会变成半角 `|`，所以两种都收。
_DSML_BAR = r"[|｜]{1,3}"
_RE_DSML_OPEN = re.compile(r"<" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"\s*([A-Za-z_]+)",
                           re.I)
_RE_DSML_TOKEN = re.compile(r"</?" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"[^>]*>", re.I)
_RE_DSML_INVOKE = re.compile(r"<" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"\s*invoke\b",
                             re.I)
_RE_DSML_PARAM = re.compile(
    r"<" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"\s*parameter\b([^>]*)>(.*?)"
    r"(?:</" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"\s*parameter\s*>"
    r"|(?=<" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r")|$)",
    re.S | re.I)
_RE_ATTR = re.compile(r"([A-Za-z_][\w-]*)\s*=\s*\"([^\"]*)\"")
_RE_DSML_BLOCK = re.compile(
    r"<" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"\s*(?:tool_calls|calls)\b[^>]*>.*?"
    r"(?:</" + _DSML_BAR + r"\s*DSML\s*" + _DSML_BAR + r"\s*(?:tool_calls|calls)\s*>|$)",
    re.S | re.I)


def has_dsml(text):
    """文本里有没有 DSML 标记（用于快速分支，不解析）。"""
    return bool(text) and bool(_RE_DSML_OPEN.search(text))


def strip_dsml(text):
    """把 DSML 标记**整块删掉**，只留下真正的人话。

    两步：① 先删掉完整的 tool_calls 块；② 再清掉残留的孤立标记。
    这样即使用户看到的是「模型乱吐 DSML」，也不会在聊天里看到乱码。
    """
    if not text:
        return text
    out = _RE_DSML_BLOCK.sub("", text)
    out = _RE_DSML_INVOKE.sub("", out)
    out = _RE_DSML_TOKEN.sub("", out)
    return out.strip()


def extract_dsml_tool_calls(text):
    """解析 DeepSeek DSML 工具标记 → 标准 OpenAI tool_calls。解析不出返回 None。

    容忍：闭合标签缺失/被截断（此时以「下一个 invoke 开始」或文末为界）。
    """
    try:
        if not has_dsml(text):
            return None
        parts = _RE_DSML_INVOKE.split(text)[1:]     # 每段以 ` name="X" ...>body` 开头
        out = []
        for i, seg in enumerate(parts):
            gt = seg.find(">")
            if gt == -1:                            # 没有 `>`，退而求其次搜属性
                attrs, body = seg, ""
            else:
                attrs, body = seg[:gt], seg[gt + 1:]
            name = ""
            for k, v in _RE_ATTR.findall(attrs):
                if k.lower() == "name":
                    name = v.strip()
                    break
            if not name:
                continue
            args = {}
            for attr_str, value in _RE_DSML_PARAM.findall(body):
                key = ""
                for k, v in _RE_ATTR.findall(attr_str):
                    if k.lower() == "name":
                        key = v.strip()
                        break
                if key:
                    args[key] = value.strip()
            out.append({"id": "call_%d" % (i + 1), "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(args, ensure_ascii=False)}})
        return out or None
    except Exception:
        return None


def filter_tool_calls(tool_calls, allowed_names):
    """只保留**客户端真的声明过**的工具调用。

    为什么必须有这一步：上游 Agent 自带的工具（`Write` / `Read` / `Bash` / `future_task` …）
    模型也会去调，但客户端根本不认识它们 —— 转发过去只会让客户端报「未知工具」。
    ⇒ 未声明的调用**直接丢弃**（连正文里的标记一起清掉，别让用户看到乱码）。

    `allowed_names` 为空/None → 不过滤，原样返回。
    返回 `(kept, dropped)` 两个列表，方便调用方记录日志。
    """
    calls = list(tool_calls or [])
    if not allowed_names:
        return calls, []
    allow = {str(n) for n in allowed_names if n}
    kept, dropped = [], []
    for c in calls:
        name = ((c or {}).get("function") or {}).get("name")
        (kept if name in allow else dropped).append(c)
    return kept, dropped


def validate_tool_calls(tool_calls):
    """校验还原出来的 tool_calls 能不能用。返回 `(ok: bool, errors: list[str])`。

    ⚠️ `arguments` 是**模型生成的字符串**，在真正执行工具前**必须**过一遍这里 ——
       一个语法错的 JSON 就足以让下游工具直接崩掉。
    """
    errors = []
    if not isinstance(tool_calls, list) or not tool_calls:
        return False, ["tool_calls 为空或不是数组"]
    for i, c in enumerate(tool_calls):
        fn = (c or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            errors.append("[%d] 缺少 function.name" % i)
        raw = fn.get("arguments", "")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw or "{}")
            except Exception as e:
                errors.append("[%d] arguments 不是合法 JSON: %s" % (i, e))
                continue
        else:
            parsed = raw
        if not isinstance(parsed, dict):
            errors.append("[%d] arguments 不是 JSON 对象（得到 %s）" % (i, type(parsed).__name__))
    return (not errors), errors


def openai_tool_response(model, tool_calls, usage=None, content=None):
    """组装**标准 OpenAI** 的非流式响应。

    客户端只要看到 `finish_reason == "tool_calls"` 就会去执行工具，不会再当正文显示。
    """
    return {
        "id": "chatcmpl-" + os.urandom(6).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }],
        "usage": usage or {},
    }


def sse_from_message(message, model):
    """把一条完整 message 包成 OpenAI 风格 SSE（客户端要 `stream:true` 时用）。

    ⚠️ 为什么必须「先收全、再合成」：模型是把 tool_calls 当**文本**吐出来的，
       流式透传时没法回头把它改写进 `delta.tool_calls` —— 只能拿到完整文本后**重放**一遍。
       代价是这一次请求失去打字机效果（见 README「局限」）。
    """
    base = {"id": "chatcmpl-" + os.urandom(6).hex(),
            "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    chunks = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant"},
                                   "finish_reason": None}])]
    if message.get("tool_calls"):
        chunks.append(dict(base, choices=[{"index": 0,
                                           "delta": {"tool_calls": message["tool_calls"]},
                                           "finish_reason": None}]))
    if message.get("content"):
        chunks.append(dict(base, choices=[{"index": 0,
                                           "delta": {"content": message["content"]},
                                           "finish_reason": None}]))
    chunks.append(dict(base, choices=[{"index": 0, "delta": {},
                                       "finish_reason": ("tool_calls" if message.get("tool_calls")
                                                         else "stop")}]))
    body = "".join("data: " + json.dumps(c, ensure_ascii=False) + "\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode("utf-8")


def to_anthropic_tool_use(tool_calls):
    """顺手转成 Anthropic Messages API 的 content block（给 Claude Code 这类客户端用）。

        OpenAI:     {"id": .., "function": {"name": .., "arguments": "<json 字符串>"}}
        Anthropic:  {"type": "tool_use", "id": .., "name": .., "input": {...}}
    """
    out = []
    for c in tool_calls or []:
        fn = (c or {}).get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw or "{}")
            except Exception:
                raw = {"_raw": raw}
        out.append({"type": "tool_use",
                    "id": c.get("id") or "",
                    "name": fn.get("name") or "",
                    "input": raw if isinstance(raw, dict) else {}})
    return out
