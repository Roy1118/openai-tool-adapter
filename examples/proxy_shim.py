#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""proxy_shim.py —— 端到端可跑的示范：一个只会聊天的上游 + 我们的适配层。

跑起来能看到三件事：
    1. **注入**：发给上游的 system 里多了工具说明
    2. **还原**：上游把调用当纯文本吐回来 → 我们还原成标准 tool_calls
    3. **清洗**：上游吐原生标记（DSML）时，用户看到的是干净正文

    python examples/proxy_shim.py            # 全跑一遍
    python examples/proxy_shim.py --serve    # 起一个真 HTTP 服务（127.0.0.1:8899）
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import (  # noqa: E402
    apply_tools_to_messages,
    extract_tool_calls,
    filter_tool_calls,
    openai_tool_response,
    strip_dsml,
    validate_tool_calls,
)


# ─────────────────────── 假装这是一个「不支持 tools」的上游 ───────────────────────

def dumb_upstream(sent_messages):
    """它只读 messages，完全不知道 tools 是什么。

    这里用一个「守规矩」的假模型：看到 system 里有工具说明 + 用户喊画图，
    就按约定吐 JSON。真实世界里这取决于模型能力。
    """
    system = next((m.get("content") for m in sent_messages
                   if m.get("role") == "system"), "") or ""
    user = next((m.get("content") for m in reversed(sent_messages)
                 if m.get("role") == "user"), "") or ""

    injected = "【可用工具】" in system and "generate_image" in system
    if injected and ("画" in user or "自拍" in user):
        return ('{"tool_calls":[{"id":"call_1","type":"function","function":'
                '{"name":"generate_image","arguments":'
                '"{\\"prompt\\":\\"银白长发少女自拍，窗边自然光\\",\\"aspect_ratio\\":\\"2:3\\"}"}}]}')
    return "在的呀，今天想聊点什么？"


def rude_upstream(sent_messages):
    """一个「不守规矩」的假模型：用自己训练过的原生格式（DSML）调内置工具。"""
    return ("我记一下。\n"
            "<｜｜DSML｜｜ calls>\n"
            "<｜｜DSML｜｜ invoke name=\"Write\">\n"
            "<｜｜DSML｜｜ parameter name=\"file_path\" string=\"true\">"
            "/tmp/memory/user_a.md</｜｜DSML｜｜ parameter>\n"
            "<｜｜DSML｜｜ parameter name=\"content\" string=\"true\">喜欢猫"
            "</｜｜DSML｜｜ parameter>\n"
            "</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>")


def handle(messages, tools, upstream, label):
    """中间层的完整流程（这就是你该抄进项目的部分）。"""
    names = [t["function"]["name"] for t in tools]
    print("\n" + "=" * 68)
    print("### %s" % label)

    sent = apply_tools_to_messages(messages, tools)
    print("  ① 注入后 system 长度: %d 字符（含工具说明 = %s）"
          % (len(sent[0]["content"] or ""), "【可用工具】" in (sent[0]["content"] or "")))

    reply = upstream(sent)
    print("  ② 上游原样返回: %s" % repr(reply[:110]))

    calls = extract_tool_calls(reply) or []
    kept, dropped = filter_tool_calls(calls, names)
    print("  ③ 解析出: %s" % [c["function"]["name"] for c in calls])
    print("     保留(客户端认识): %s" % [c["function"]["name"] for c in kept])
    print("     丢弃(上游自带)  : %s" % [c["function"]["name"] for c in dropped])

    if kept:
        ok, errors = validate_tool_calls(kept)
        print("     校验: %s" % ("通过" if ok else errors))
        resp = openai_tool_response("demo-model", kept)
        print("  ➜ 返回给客户端: finish_reason=%s, tool_calls=%s"
              % (resp["choices"][0]["finish_reason"],
                 json.dumps([{ "name": c["function"]["name"],
                               "args": json.loads(c["function"]["arguments"])}
                             for c in kept], ensure_ascii=False)))
    else:
        clean = strip_dsml(reply)
        print("  ➜ 返回给客户端: 纯文本 %r" % clean)
        if clean != reply:
            print("     （原文里的工具标记已被清掉，用户不会看到乱码）")


# ─────────────────────── 也可以起成一个真服务 ───────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            body = {}
        out = handle_once(body)
        data = json.dumps(out, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def handle_once(body):
    """真正跑在服务里的中间层 —— 可以直接拿去接你的客户端。"""
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    model = body.get("model") or "demo-model"

    names = [t["function"]["name"] for t in tools]
    sent = apply_tools_to_messages(messages, tools)
    reply = dumb_upstream(sent)                 # ← 换成你那个「不支持 tools」的上游

    calls = extract_tool_calls(reply) or []
    kept, _dropped = filter_tool_calls(calls, names)
    if kept:
        return openai_tool_response(model, kept)
    return {
        "id": "chatcmpl-shim", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": strip_dsml(reply)}}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="起 HTTP 服务（默认只跑自测）")
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args()

    if a.serve:
        print("中间层已启动：http://127.0.0.1:%d/v1/chat/completions" % a.port)
        print("试一下：")
        print('''  curl -s http://127.0.0.1:%d/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"demo","messages":[{"role":"user","content":"画一张自拍"}],
         "tools":[{"type":"function","function":{"name":"generate_image","parameters":{"type":"object"}}}]}'
''' % a.port)
        HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
        return

    tools = [{"type": "function", "function": {
        "name": "generate_image",
        "description": "使用生图模型生成图片；用户要求自拍/头像/人像时应填写 persona。",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "生图提示词。"},
            "persona": {"type": "string", "description": "人设名称。"},
            "aspect_ratio": {"type": "string", "enum": ["1:1", "2:3", "3:4", "16:9"]}},
            "required": []}}}]

    handle([{"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": "帮我画一张自拍"}],
           tools, dumb_upstream, "场景 A：模型守规矩 → 还原成标准 tool_calls")

    handle([{"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": "记住我喜欢猫"}],
           tools, rude_upstream, "场景 B：模型用原生格式调内置工具 → 过滤掉 + 清干净正文")

    handle([{"role": "user", "content": "在干嘛"}], [], dumb_upstream,
           "场景 C：不带工具 → 完全没有适配介入，就是普通聊天")
    print("\n" + "=" * 68)


if __name__ == "__main__":
    main()
