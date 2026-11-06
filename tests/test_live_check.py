import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.llm import ModelError
from scripts.live_check import main
from tests.test_agent import ScriptModel, call, final


class LiveCheckTests(unittest.TestCase):
    def run_check(self, model):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                with patch.object(sys, "argv", ["live_check"]), patch.dict(os.environ, {
                    "LLM_API_KEY": "test-only-placeholder", "LLM_MODEL": "test-model",
                }), patch("scripts.live_check.ChatModel", return_value=model), contextlib.redirect_stdout(io.StringIO()):
                    code = main()
                    self.assertNotIn("LLM_API_KEY", os.environ)
                report = Path("data/live_check.json").read_text(encoding="utf-8")
                self.assertNotIn("test-only-placeholder", report)
                return code, json.loads(report)
            finally:
                os.chdir(previous)

    def test_success_report_requires_actual_tool_state(self):
        model = ScriptModel(
            final("你好"), final("记住了"), final("你叫小林"),
            call("calculator", {"expression": "18*7"}, "c1"),
            call("todo", {"action": "add", "text": "复习计算结果126"}, "c2"), final(),
            call("todo", {"action": "done", "item_id": 1}, "c3"), final(),
            call("todo", {"action": "list"}, "c4"), final(),
            call("todo", {"action": "list"}, "c5"), final(),
            call("search", {"query": "Agent runtime"}, "c6"), final("这是模拟数据"),
        )
        code, report = self.run_check(model)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["checks"]), 8)
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        self.assertEqual(model.key, "")

    def test_model_failure_cannot_produce_passing_report(self):
        code, report = self.run_check(ScriptModel(ModelError("模型网络请求失败或超时")))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("超时", report["error"])


if __name__ == "__main__":
    unittest.main()
