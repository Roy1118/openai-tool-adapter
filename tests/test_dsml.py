#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSML（DeepSeek 原生工具标记）解析 / 过滤 / 清洗的专项测试。

    python tests/test_dsml.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import (  # noqa: E402
    extract_dsml_tool_calls,
    extract_tool_calls,
    filter_tool_calls,
    has_dsml,
    render_tools,
    strip_dsml,
    apply_tools_to_messages,
    TOOL_INSTRUCTION,
)

# 线上真实抓到的形状（QQ 群里她吐出来的原文，见 2026-10-16 事故）
REAL_WRITE = (
    "既然你当面问了，我顺手把你的名字记小本本了。\n"
    "<｜｜DSML｜｜ calls>\n"
    "<｜｜DSML｜｜ invoke name=\"Write\">\n"
    "<｜｜DSML｜｜ parameter name=\"file_path\" string=\"true\">"
    "/home/admin/.codebuddy/projects/x/memory/user_a.md</｜｜DSML｜｜ parameter>\n"
    "<｜｜DSML｜｜ parameter name=\"content\" string=\"true\">---\nname: 车车壬\n"
    "type: user\n---\n</｜｜DSML｜｜ parameter>\n"
    "</｜｜DSML｜｜ invoke>\n"
    "</｜｜DSML｜｜ calls>"
)
REAL_FUTURE = (
    "<｜｜DSML｜｜ calls>\n"
    "<｜｜DSML｜｜ invoke name=\"future_task\">\n"
    "<｜｜DSML｜｜ parameter name=\"action\" string=\"true\">create</｜｜DSML｜｜ parameter>\n"
    "<｜｜DSML｜｜ parameter name=\"prompt\" string=\"true\">明天提醒他早睡</｜｜DSML｜｜ parameter>\n"
    "</｜｜DSML｜｜ invoke>\n"
    "</｜｜DSML｜｜ calls>"
)


class DetectTest(unittest.TestCase):
    def test_has_dsml_true(self):
        self.assertTrue(has_dsml(REAL_WRITE))
        self.assertTrue(has_dsml(REAL_FUTURE))

    def test_has_dsml_false(self):
        for t in (None, "", "普通聊天", '{"tool_calls": []}', "＜＜DSML＞＞"):
            self.assertFalse(has_dsml(t), repr(t))

    def test_half_width_bars_also_accepted(self):
        """有些转发/字体把全角竖线变成半角 —— 两种都要认。"""
        self.assertTrue(has_dsml('<||DSML||invoke name="Write">'))


class ParseTest(unittest.TestCase):
    def test_parse_write(self):
        calls = extract_dsml_tool_calls(REAL_WRITE)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        fn = calls[0]["function"]
        self.assertEqual(fn["name"], "Write")
        args = json.loads(fn["arguments"])
        self.assertIn("user_a.md", args["file_path"])
        self.assertIn("车车壬", args["content"])
        self.assertEqual(calls[0]["type"], "function")
        self.assertTrue(calls[0]["id"])

    def test_parse_two_invokes(self):
        text = ('<｜｜DSML｜｜ calls>\n'
                '<｜｜DSML｜｜ invoke name="Read">\n'
                '<｜｜DSML｜｜ parameter name="file_path" string="true">/tmp/a</｜｜DSML｜｜ parameter>\n'
                '</｜｜DSML｜｜ invoke>\n'
                '<｜｜DSML｜｜ invoke name="Bash">\n'
                '<｜｜DSML｜｜ parameter name="command" string="true">ls -la</｜｜DSML｜｜ parameter>\n'
                '</｜｜DSML｜｜ invoke>\n'
                '</｜｜DSML｜｜ calls>')
        calls = extract_dsml_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["Read", "Bash"])
        self.assertEqual(json.loads(calls[1]["function"]["arguments"])["command"], "ls -la")

    def test_parse_without_closing_tags(self):
        """真实场景里响应常被截断 —— 闭合标签缺失也要能解析。"""
        text = ('<｜｜DSML｜｜ calls>\n'
                '<｜｜DSML｜｜ invoke name="Write">\n'
                '<｜｜DSML｜｜ parameter name="file_path" string="true">/tmp/x</｜｜DSML｜｜ parameter>')
        calls = extract_dsml_tool_calls(text)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "Write")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["file_path"], "/tmp/x")

    def test_no_parameters(self):
        text = ('<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="ping">'
                '</｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>')
        calls = extract_dsml_tool_calls(text)
        self.assertEqual(calls[0]["function"]["name"], "ping")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {})

    def test_values_with_newlines_and_chinese(self):
        calls = extract_dsml_tool_calls(REAL_WRITE)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertIn("\n", args["content"])
        self.assertIn("车车壬", args["content"])

    def test_not_dsml_returns_none(self):
        for t in (None, "", "普通聊天", "<｜｜DSML｜｜ calls>",
                  "<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke></｜｜DSML｜｜ invoke>"):  # 缺 name
            self.assertIsNone(extract_dsml_tool_calls(t), repr(t))

    def test_extract_tool_calls_falls_back_to_dsml(self):
        """主入口要能自动认出 DSML（不需要调用方判断用哪种协议）。"""
        calls = extract_tool_calls(REAL_FUTURE)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "future_task")

    def test_json_takes_priority(self):
        """两种格式同时出现时，认我们注入的 JSON（客户端工具优先）。"""
        text = ('{"tool_calls":[{"id":"c1","function":{"name":"generate_image","arguments":"{}"}}]}\n'
                + REAL_FUTURE)
        calls = extract_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["generate_image"])


class StripTest(unittest.TestCase):
    def test_keeps_human_text(self):
        out = strip_dsml(REAL_WRITE)
        self.assertIn("我顺手把你的名字记小本本了", out)
        self.assertNotIn("DSML", out)
        self.assertNotIn("user_a.md", out)

    def test_pure_markup_becomes_empty(self):
        self.assertEqual(strip_dsml(REAL_FUTURE), "")

    def test_normal_text_untouched(self):
        for t in ("在干嘛呀", "**加粗** 和 `代码`", ""):
            self.assertEqual(strip_dsml(t), t)

    def test_no_leftover_fragments(self):
        """清完不能留下半个标记 —— 这是用户实际看到乱码的原因。"""
        for text in (REAL_WRITE, REAL_FUTURE, REAL_WRITE + "\n还有一句",
                     '<｜｜DSML｜｜invoke name="X">y</｜｜DSML｜｜invoke>'):
            out = strip_dsml(text)
            self.assertNotIn("｜", out, "残留了全角竖线: %r" % out)
            self.assertNotIn("DSML", out, "残留了 DSML: %r" % out)
            self.assertNotIn("invoke", out, "残留了 invoke: %r" % out)


class FilterTest(unittest.TestCase):
    def test_drops_builtin_tools(self):
        calls = extract_tool_calls(REAL_WRITE + REAL_FUTURE)
        kept, dropped = filter_tool_calls(calls, ["generate_image", "send_message_to_user"])
        self.assertEqual(kept, [])
        self.assertEqual([c["function"]["name"] for c in dropped], ["Write", "future_task"])

    def test_keeps_declared_tools(self):
        calls = [{"id": "c1", "type": "function",
                  "function": {"name": "generate_image", "arguments": "{}"}},
                 {"id": "c2", "type": "function",
                  "function": {"name": "Bash", "arguments": "{}"}}]
        kept, dropped = filter_tool_calls(calls, ["generate_image"])
        self.assertEqual([c["function"]["name"] for c in kept], ["generate_image"])
        self.assertEqual([c["function"]["name"] for c in dropped], ["Bash"])

    def test_no_whitelist_passes_through(self):
        calls = [{"id": "c", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}]
        for allow in (None, [], set()):
            kept, dropped = filter_tool_calls(calls, allow)
            self.assertEqual(len(kept), 1)
            self.assertEqual(dropped, [])

    def test_empty_input(self):
        self.assertEqual(filter_tool_calls(None, ["a"]), ([], []))


class RealIncidentTest(unittest.TestCase):
    """复现 2026-10-16 线上事故：注入生效了，但模型吐 DSML 给用户看。"""

    def test_before_fix_garbage_was_visible(self):
        # 修之前：extract 只认 JSON → 返回 None → 网关把原文（含 DSML）当正文发出去
        self.assertNotIn("DSML", json.dumps({"c": strip_dsml(REAL_WRITE)}))
        # 修之后：既能还原成 tool_calls，又能把标记清干净
        self.assertIsNotNone(extract_tool_calls(REAL_WRITE))
        self.assertNotIn("DSML", strip_dsml(REAL_WRITE))

    def test_incident_end_to_end(self):
        """完整走一遍：模型吐 DSML(内置工具) → 过滤掉 → 只把正文还给人。"""
        raw = REAL_WRITE
        calls = extract_tool_calls(raw) or []
        kept, dropped = filter_tool_calls(calls, ["generate_image"])
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)
        text = strip_dsml(raw)          # 客户端应看到这个
        self.assertEqual(text, "既然你当面问了，我顺手把你的名字记小本本了。")

    def test_declared_tool_via_dsml_still_works(self):
        """即使模型用 DSML 格式调客户端工具，也要能正常还原并保留。"""
        text = ('<｜｜DSML｜｜ calls>\n'
                '<｜｜DSML｜｜ invoke name="generate_image">\n'
                '<｜｜DSML｜｜ parameter name="prompt" string="true">自拍，微笑</｜｜DSML｜｜ parameter>\n'
                '<｜｜DSML｜｜ parameter name="aspect_ratio" string="true">2:3</｜｜DSML｜｜ parameter>\n'
                '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
        calls = extract_tool_calls(text)
        kept, dropped = filter_tool_calls(calls, ["generate_image"])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])
        self.assertEqual(json.loads(kept[0]["function"]["arguments"]),
                         {"prompt": "自拍，微笑", "aspect_ratio": "2:3"})


class InstructionTest(unittest.TestCase):
    def test_warns_no_builtin_tools(self):
        """注入指令必须明确否认内置工具 —— 否则模型会用原生协议（本次事故根因）。"""
        self.assertIn("没有任何内置工具", TOOL_INSTRUCTION)
        self.assertIn("DSML", TOOL_INSTRUCTION)

    def test_injected_prompt_carries_the_warning(self):
        out = apply_tools_to_messages([{"role": "user", "content": "x"}],
                                      [{"type": "function",
                                        "function": {"name": "generate_image"}}])
        self.assertIn("没有任何内置工具", out[0]["content"])
        self.assertIn("generate_image", out[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
