#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零依赖测试：直接 `python tests/test_adapter.py` 就能跑（不需要 pytest）。

    python tests/test_adapter.py          # 全跑
    python tests/test_adapter.py -v       # 看每条用例名
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import (  # noqa: E402
    TOOL_INSTRUCTION,
    apply_tools_to_messages,
    extract_tool_calls,
    openai_tool_response,
    render_tools,
    sse_from_message,
    to_anthropic_tool_use,
    validate_tool_calls,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询某城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}

IMAGE = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "生成图片",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "提示词"},
                "persona": {"type": "string", "description": "人设名"},
                "aspect_ratio": {"type": "string", "enum": ["1:1", "2:3", "3:4", "16:9"]},
            },
            "required": ["prompt"],
        },
    },
}


class RenderToolsTest(unittest.TestCase):
    def test_basic_fields(self):
        out = render_tools([WEATHER])
        self.assertIn("get_weather", out)
        self.assertIn("查询某城市的当前天气", out)
        self.assertIn("city:城市名，如 北京[必填]", out)
        self.assertIn("unit(c/f)", out)          # enum 展开

    def test_empty_inputs(self):
        for v in (None, [], "not-a-list", 0, {}):
            self.assertEqual(render_tools(v), "")

    def test_bare_function_object(self):
        """裸 function 对象（没有外层 type/function 包装）也要认。"""
        out = render_tools([{"name": "ping", "description": "pong"}])
        self.assertIn("ping", out)
        self.assertIn("(无)", out)                # 没有参数

    def test_skips_nameless_and_bad_types(self):
        self.assertEqual(render_tools([{"description": "没有名字"}]), "")
        self.assertEqual(render_tools(["字符串", 123, None]), "")

    def test_compact_than_raw_json(self):
        """「紧凑」不是口号：渲染结果必须明显短于原样 JSON。"""
        tools = [WEATHER, IMAGE]
        raw = len(json.dumps(tools, ensure_ascii=False))
        rendered = len(render_tools(tools))
        self.assertLess(rendered, raw * 0.6, "渲染 %d 字符 vs 原始 %d 字符" % (rendered, raw))

    def test_description_truncated(self):
        long_desc = "很长的描述" * 200
        t = {"type": "function", "function": {"name": "x", "description": long_desc}}
        self.assertLess(len(render_tools([t])), 320)


class ApplyToolsTest(unittest.TestCase):
    def test_appends_to_existing_system(self):
        msgs = [{"role": "system", "content": "你是助手。"},
                {"role": "user", "content": "北京天气"}]
        out = apply_tools_to_messages(msgs, [WEATHER])
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["content"].startswith("你是助手。"))
        self.assertIn("get_weather", out[0]["content"])
        self.assertIn("只输出一行 JSON", out[0]["content"])

    def test_creates_system_when_absent(self):
        msgs = [{"role": "user", "content": "北京天气"}]
        out = apply_tools_to_messages(msgs, [WEATHER])
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("get_weather", out[0]["content"])
        self.assertEqual(out[1]["role"], "user")

    def test_does_not_mutate_input(self):
        msgs = [{"role": "system", "content": "原始"}, {"role": "user", "content": "hi"}]
        snapshot = json.dumps(msgs, ensure_ascii=False)
        apply_tools_to_messages(msgs, [WEATHER])
        self.assertEqual(json.dumps(msgs, ensure_ascii=False), snapshot)

    def test_no_tools_returns_copy(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = apply_tools_to_messages(msgs, [])
        self.assertEqual(out, msgs)
        self.assertIsNot(out, msgs)
        self.assertIsNot(out[0], msgs[0])

    def test_non_string_system_content(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "块式"}]}]
        out = apply_tools_to_messages(msgs, [WEATHER])
        self.assertIsInstance(out[0]["content"], str)
        self.assertIn("get_weather", out[0]["content"])

    def test_instruction_placeholder_not_formatted(self):
        """正文里必须保留 {tools} 之外的**字面**花括号（JSON 示例）。"""
        out = apply_tools_to_messages([{"role": "user", "content": "x"}], [WEATHER])
        self.assertIn('"tool_calls":[{"id":"call_1"', out[0]["content"])
        self.assertNotIn("{tools}", out[0]["content"])

    def test_instruction_example_is_valid_json(self):
        """⭐ 回归：提示词里给模型的 JSON 示例必须是**能解析的真 JSON**。

        踩过的坑：示例写成双层花括号（`{{"tool_calls"...`）—— 功能表面能用，
        但模型看到的样例是非法 JSON，小模型容易照抄出错。
        """
        self.assertNotIn("{{", TOOL_INSTRUCTION, "示例里出现了双层花括号（非法 JSON）")
        line = next(l for l in TOOL_INSTRUCTION.split("\n")
                    if l.startswith('{"tool_calls"'))
        data = json.loads(line)                       # 能解析才算过
        call = data["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "工具名")
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"参数": "值"})


class ExtractToolCallsTest(unittest.TestCase):
    def test_clean_json(self):
        text = ('{"tool_calls":[{"id":"call_1","type":"function","function":'
                '{"name":"get_weather","arguments":"{\\"city\\":\\"北京\\"}"}}]}')
        calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "北京"})
        self.assertEqual(calls[0]["type"], "function")

    def test_fenced_code_block(self):
        text = ('```json\n{"tool_calls":[{"id":"call_1","function":'
                '{"name":"f","arguments":"{}"}}]}\n```')
        self.assertEqual(len(extract_tool_calls(text)), 1)

    def test_plain_fence_without_language(self):
        text = '```\n{"tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}\n```'
        self.assertEqual(len(extract_tool_calls(text)), 1)

    def test_surrounding_prose(self):
        text = ('好的，我来查一下：\n'
                '{"tool_calls":[{"id":"c1","function":{"name":"f","arguments":"{}"}}]}\n'
                '稍等～')
        self.assertEqual(len(extract_tool_calls(text)), 1)

    def test_multiple_calls(self):
        text = ('{"tool_calls":[{"id":"a","function":{"name":"f1","arguments":"{}"}},'
                '{"id":"b","function":{"name":"f2","arguments":"{\\"x\\":1}"}}]}')
        calls = extract_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["f1", "f2"])

    def test_arguments_as_object_becomes_string(self):
        text = '{"tool_calls":[{"function":{"name":"f","arguments":{"city":"北京"}}}]}'
        calls = extract_tool_calls(text)
        self.assertIsInstance(calls[0]["function"]["arguments"], str)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "北京"})

    def test_nested_braces_inside_arguments_string(self):
        """字符串里的花括号不能被当成 JSON 边界 —— 这是最容易写错的地方。"""
        text = ('{"tool_calls":[{"function":{"name":"f",'
                '"arguments":"{\\"a\\":{\\"b\\":{\\"c\\":1}}}"}}]}')
        calls = extract_tool_calls(text)
        self.assertIsNotNone(calls)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"a": {"b": {"c": 1}}})

    def test_missing_id_gets_generated(self):
        calls = extract_tool_calls('{"tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}')
        self.assertEqual(calls[0]["id"], "call_1")

    def test_not_tool_calls_returns_none(self):
        for text in (None, "", "   ", "今天天气不错", "{}", '{"a":1}',
                     '{"tool_calls":[]}', '{"tool_calls":"不是数组"}',
                     '{"tool_calls":[{"function":{}}]}',        # 缺 name
                     '{"tool_calls":[123]}',                     # 元素不是对象
                     '{"tool_calls":[{"function":{"name":"f"}}]',  # JSON 截断
                     "{不是合法 JSON}"):
            self.assertIsNone(extract_tool_calls(text), "不应解析出: %r" % (text,))

    def test_keyword_mentioned_but_no_json(self):
        """正文里提到 tool_calls 但没给 JSON —— 不能误判成工具调用。"""
        self.assertIsNone(extract_tool_calls("我不需要调用工具，不使用 tool_calls。"))


class ValidateToolCallsTest(unittest.TestCase):
    def test_ok(self):
        calls = extract_tool_calls('{"tool_calls":[{"function":{"name":"f","arguments":"{\\"a\\":1}"}}]}')
        ok, errs = validate_tool_calls(calls)
        self.assertTrue(ok, errs)
        self.assertEqual(errs, [])

    def test_bad_json_arguments(self):
        calls = [{"id": "c", "type": "function",
                  "function": {"name": "f", "arguments": "{不是 JSON"}}]
        ok, errs = validate_tool_calls(calls)
        self.assertFalse(ok)
        self.assertIn("不是合法 JSON", errs[0])

    def test_arguments_not_object(self):
        calls = [{"id": "c", "function": {"name": "f", "arguments": "[1,2]"}}]
        ok, errs = validate_tool_calls(calls)
        self.assertFalse(ok)
        self.assertIn("不是 JSON 对象", errs[0])

    def test_empty_and_none(self):
        self.assertFalse(validate_tool_calls([])[0])
        self.assertFalse(validate_tool_calls(None)[0])


class ResponseShapeTest(unittest.TestCase):
    def test_openai_response(self):
        calls = [{"id": "c1", "type": "function",
                  "function": {"name": "f", "arguments": "{}"}}]
        r = openai_tool_response("my-model", calls, {"prompt_tokens": 1})
        self.assertEqual(r["object"], "chat.completion")
        self.assertEqual(r["model"], "my-model")
        self.assertEqual(r["usage"], {"prompt_tokens": 1})
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertEqual(ch["message"]["tool_calls"], calls)
        self.assertIsNone(ch["message"]["content"])

    def test_sse_contains_tool_calls_and_done(self):
        msg = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "f", "arguments": "{}"}}]}
        raw = sse_from_message(msg, "my-model").decode("utf-8")
        lines = [l for l in raw.split("\n") if l.startswith("data: ")]
        payloads = [json.loads(l[6:]) for l in lines if l[6:].strip() != "[DONE]"]
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        self.assertEqual(payloads[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertIn("tool_calls", payloads[1]["choices"][0]["delta"])
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_sse_plain_chat_finish_reason_stop(self):
        raw = sse_from_message({"role": "assistant", "content": "你好"},
                               "m").decode("utf-8")
        payloads = [json.loads(l[6:]) for l in raw.split("\n")
                    if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payloads[-1]["choices"][0]["delta"], {})

    def test_anthropic_mapping(self):
        calls = [{"id": "c1", "type": "function",
                  "function": {"name": "generate_image",
                               "arguments": '{"prompt":"自拍","aspect_ratio":"2:3"}'}}]
        blocks = to_anthropic_tool_use(calls)
        self.assertEqual(blocks[0]["type"], "tool_use")
        self.assertEqual(blocks[0]["name"], "generate_image")
        self.assertEqual(blocks[0]["input"], {"prompt": "自拍", "aspect_ratio": "2:3"})

    def test_anthropic_bad_arguments_wrapped(self):
        blocks = to_anthropic_tool_use([{"id": "c", "function": {"name": "f", "arguments": "{坏"}}])
        self.assertEqual(blocks[0]["input"], {"_raw": "{坏"})


class RoundTripTest(unittest.TestCase):
    """完整链路：注入 → 模拟上游把它当文本吐回来 → 还原 → 校验 → 组装响应。"""

    def test_full_round_trip(self):
        tools = [IMAGE]
        messages = [{"role": "system", "content": "你是画师。"},
                    {"role": "user", "content": "画一张你的自拍"}]

        sent = apply_tools_to_messages(messages, tools)
        self.assertIn("generate_image", sent[0]["content"])
        self.assertIn("persona", sent[0]["content"])   # 参数说明确实到了系统提示词里

        # 模拟一个「收到注入提示词后照做」的上游
        upstream_reply = (
            "```json\n"
            '{"tool_calls":[{"id":"call_1","type":"function","function":'
            '{"name":"generate_image","arguments":'
            '"{\\"prompt\\":\\"自拍，微笑\\",\\"aspect_ratio\\":\\"2:3\\"}"}}]}\n'
            "```"
        )
        calls = extract_tool_calls(upstream_reply)
        self.assertIsNotNone(calls)

        ok, errs = validate_tool_calls(calls)
        self.assertTrue(ok, errs)

        resp = openai_tool_response("demo-model", calls, {"total_tokens": 42})
        fn = resp["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(fn["name"], "generate_image")
        self.assertEqual(json.loads(fn["arguments"])["aspect_ratio"], "2:3")

    def test_plain_chat_not_hijacked(self):
        """不带工具、或模型正常聊天时，绝不能误判成工具调用。"""
        messages = [{"role": "user", "content": "在干嘛"}]
        sent = apply_tools_to_messages(messages, [])       # 没工具 → 不注入
        self.assertNotIn("tool_calls", json.dumps(sent, ensure_ascii=False))
        self.assertIsNone(extract_tool_calls("在等你派活儿呀～"))

    def test_instruction_has_all_placeholders(self):
        self.assertIn("{tools}", TOOL_INSTRUCTION)
        self.assertIn("tool_calls", TOOL_INSTRUCTION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
