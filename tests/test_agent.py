import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from agent.context import ContextBuilder, ContextOverflow
from agent.llm import ChatModel, MockModel, ModelError, parse_message
from agent.runtime import Runtime
from agent.store import Store, BusyError
from agent.tools import ToolContext, make_registry


class ScriptModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.contexts = []

    def complete(self, messages, tools):
        self.contexts.append((messages, tools))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def final(text="完成"):
    return {"role": "assistant", "content": text}


def call(name, args, cid="call_1"):
    return {"role": "assistant", "content": "调用工具获取事实。", "tool_calls": [{
        "id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
    }]}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.db"
        self.store = Store(self.path)
        self.registry = make_registry()

    def runtime(self, model=None, **kwargs):
        return Runtime(self.store, model or MockModel(), self.registry, **kwargs)

    def test_direct_and_plain_followup(self):
        runtime = self.runtime()
        runtime.chat("a", "1", "我叫小林")
        self.assertIn("小林", runtime.chat("a", "1", "我叫什么")["answer"])
        self.assertNotIn("小林", runtime.chat("a", "2", "我叫什么")["answer"])

    def test_tool_followup_after_restart(self):
        self.runtime().chat("a", "1", "计算 2+3")
        restored = Runtime(Store(self.path), MockModel(), make_registry())
        answer = restored.chat("a", "1", "再乘以 4")["answer"]
        self.assertIn('"value": 20', answer)

    def test_session_and_user_isolation(self):
        runtime = self.runtime()
        runtime.chat("a", "1", "记待办 买书")
        self.assertIn("买书", runtime.chat("a", "1", "待办列表")["answer"])
        for user, sid in [("a", "2"), ("b", "1")]:
            self.assertNotIn("买书", runtime.chat(user, sid, "待办列表")["answer"])

    def test_todo_done_and_cross_session_id(self):
        runtime = self.runtime()
        runtime.chat("a", "1", "记待办 买书")
        self.assertIn('"done": 1', runtime.chat("a", "1", "完成待办 1")["answer"])
        self.assertIn('"ok": false', runtime.chat("a", "2", "完成待办 1")["answer"])

    def test_tool_schema_is_sent_and_two_step_dependency(self):
        first = call("calculator", {"expression": "2+3"})
        second = call("todo", {"action": "add", "text": "计算结果为 5"}, "call_2")
        model = ScriptModel(first, second, final())
        self.runtime(model).chat("a", "1", "计算后记待办")
        messages, schemas = model.contexts[1]
        self.assertEqual({t["function"]["name"] for t in schemas}, {"calculator", "search", "todo"})
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(json.loads(messages[-1]["content"])["data"]["value"], 5)
        self.assertIn("计算结果为 5", self.runtime().chat("a", "1", "待办列表")["answer"])

    def test_multiple_calls_are_all_paired(self):
        decision = call("calculator", {"expression": "1+1"})
        decision["tool_calls"] += call("search", {"query": "agent"}, "call_2")["tool_calls"]
        self.runtime(ScriptModel(decision, final())).chat("a", "1", "test")
        results = [m for _, m in self.store.history("a", "1") if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in results], ["call_1", "call_2"])
        self.assertTrue(json.loads(results[1]["content"])["data"]["mock"])

    def test_invalid_schema_and_unknown_tool_are_recoverable(self):
        for name, args, error in [("todo", {"action": "add"}, "invalid_arguments"),
                                  ("todo", {"action": "done", "item_id": "1"}, "invalid_arguments"),
                                  ("search", {"query": "x", "extra": 1}, "invalid_arguments"),
                                  ("missing", {}, "unknown_tool")]:
            with self.subTest(name=name, args=args):
                sid = str(args) + name
                self.runtime(ScriptModel(call(name, args), final())).chat("a", sid, "test")
                result = self.store.history("a", sid)[2][1]
                self.assertEqual(json.loads(result["content"])["error"], error)

    def test_bad_json_can_be_corrected_by_next_model_step(self):
        decision = call("calculator", {})
        decision["tool_calls"][0]["function"]["arguments"] = "{"
        model = ScriptModel(decision, call("calculator", {"expression": "3*7"}, "call_2"), final("21"))
        self.assertEqual(self.runtime(model).chat("a", "1", "test")["answer"], "21")
        self.assertIn("invalid_json", model.contexts[1][0][-1]["content"])

    def test_calculator_rejects_code_and_expensive_operations(self):
        runtime = self.runtime()
        for expression in ["__import__('os').getcwd()", "2**100000000", "1/0", "1e309", "True+1", "[1]*100"]:
            with self.subTest(expression=expression):
                self.assertIn('"ok": false', runtime.chat("a", "1", "计算 " + expression)["answer"])

    def test_step_limit_stops_loop(self):
        model = ScriptModel(call("search", {"query": "x"}), call("search", {"query": "x"}, "call_2"))
        result = self.runtime(model, max_steps=2).chat("a", "1", "test")
        self.assertIn("步数上限", result["answer"])
        self.assertEqual(len(model.contexts), 2)
        self.assertEqual(self.store.history("a", "1")[-1][1]["role"], "assistant")

    def test_model_failure_is_saved_and_lease_released(self):
        result = self.runtime(ScriptModel(ModelError("模型超时"))).chat("a", "1", "test")
        self.assertIn("超时", result["answer"])
        self.assertIn("已收到", self.runtime().chat("a", "1", "继续")["answer"])

    def test_busy_session_other_session_can_continue(self):
        owner = self.store.claim("a", "1")
        with self.assertRaises(BusyError):
            self.runtime().chat("a", "1", "test")
        self.runtime().chat("a", "2", "hello")
        self.store.release("a", "1", owner)
        self.runtime().chat("a", "1", "hello")

    def test_real_concurrent_claim_only_one_wins(self):
        barrier = threading.Barrier(2)
        results = []

        def worker():
            barrier.wait()
            try:
                results.append(self.store.claim("a", "1"))
            except BusyError:
                results.append("busy")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results.count("busy"), 1)

    def test_expired_owner_cannot_write_or_release_new_owner(self):
        old = self.store.claim("a", "1")
        with self.store.connect() as db:
            db.execute("UPDATE sessions SET lease=0")
        new = self.store.claim("a", "1")
        self.store.release("a", "1", old)
        with self.assertRaises(BusyError):
            self.store.append("a", "1", old, final())
        with self.assertRaises(BusyError):
            self.store.todo("a", "1", old, "c", "add", "unsafe")
        with self.assertRaises(BusyError):
            self.store.claim("a", "1")
        self.store.release("a", "1", new)

    def test_crash_repair_does_not_replay_todo(self):
        owner = self.store.claim("a", "1")
        self.store.append("a", "1", owner, {"role": "user", "content": "记待办"})
        self.store.append("a", "1", owner, call("todo", {"action": "add", "text": "已落库"}))
        self.store.todo("a", "1", owner, "call_1", "add", "已落库")
        self.store.release("a", "1", owner)
        answer = self.runtime().chat("a", "1", "待办列表")["answer"]
        self.assertEqual(answer.count("已落库"), 1)
        history = self.store.history("a", "1")
        self.assertEqual(json.loads(history[2][1]["content"])["error"], "interrupted")

    def test_todo_same_call_id_is_idempotent(self):
        owner = self.store.claim("a", "1")
        for _ in range(2):
            result = self.store.todo("a", "1", owner, "c", "add", "买书")
        self.assertEqual(len(result["items"]), 1)

    def test_compression_preserves_pairs_and_full_history(self):
        runtime = self.runtime(context_budget=6500)
        for _ in range(12):
            runtime.chat("a", "1", "计算 2+3")
        owner = self.store.claim("a", "1")
        messages = runtime.context.build("a", "1", owner, self.registry.schemas())
        summary, until = self.store.memory("a", "1")
        self.assertTrue(summary and until > 0)
        self.assertEqual(len(self.store.history("a", "1")), 48)
        calls = {c["id"] for m in messages for c in m.get("tool_calls", [])}
        results = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
        self.assertEqual(calls, results)
        self.assertEqual(json.loads(messages[-2]["content"])["data"]["value"], 5)
        self.store.release("a", "1", owner)
        # Summary + watermark are persistent and do not leak to a different session.
        self.assertEqual(Store(self.path).memory("a", "1"), (summary, until))
        self.assertEqual(Store(self.path).memory("a", "2"), ("", 0))

    def test_single_turn_overflow_is_reported(self):
        answer = self.runtime(context_budget=100).chat("a", "1", "hello")["answer"]
        self.assertIn("上下文过长", answer)

    def test_200_turns_then_recent_followup(self):
        runtime = self.runtime(context_budget=6500)
        for number in range(199):
            runtime.chat("a", "long", f"第 {number} 轮：复习工具循环与上下文管理。")
        runtime.chat("a", "long", "我叫小林")
        self.assertIn("小林", runtime.chat("a", "long", "我叫什么")["answer"])
        self.assertEqual(len(self.store.history("a", "long")), 402)
        self.assertGreater(self.store.memory("a", "long")[1], 0)

    def test_trace_contains_tool_correlation(self):
        result = self.runtime().chat("a", "1", "计算 1+2")
        events = self.store.traces("a", "1")
        self.assertEqual([e["event"] for e in events], ["start", "decision", "tool", "finish"])
        self.assertEqual({e["run"] for e in events}, {result["run_id"]})
        self.assertTrue(json.loads(events[2]["detail"])["call_id"])

    def test_reused_model_call_id_is_rejected(self):
        model = ScriptModel(call("search", {"query": "x"}), call("search", {"query": "x"}))
        self.assertIn("重复使用", self.runtime(model).chat("a", "1", "test")["answer"])

    def test_input_limits(self):
        for text in ["", "   ", "x" * 4001]:
            with self.assertRaises(ValueError):
                self.runtime().chat("a", "1", text)

    def test_large_tool_result_is_bounded_json(self):
        self.registry.register("large", "large", {"type": "object"}, lambda ctx: {"value": "x" * 10000})
        model = ScriptModel(call("large", {}), final())
        self.runtime(model).chat("a", "1", "test")
        result = model.contexts[1][0][-1]["content"]
        self.assertTrue(json.loads(result)["truncated"])
        self.assertLess(len(result), 2500)

    def test_unexpected_tool_failure_is_sanitized(self):
        def broken(ctx):
            raise RuntimeError("internal-secret")
        self.registry.register("broken", "broken", {"type": "object"}, broken)
        model = ScriptModel(call("broken", {}), final())
        self.runtime(model).chat("a", "1", "test")
        result = model.contexts[1][0][-1]["content"]
        self.assertIn("tool_failed", result)
        self.assertNotIn("internal-secret", result)


class ProtocolTests(unittest.TestCase):
    def test_parser_rejects_invalid_responses(self):
        bad = [{}, {"choices": []}, {"choices": [{"finish_reason": "length", "message": final()}]},
               {"choices": [{"finish_reason": "stop", "message": []}]},
               {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}]
        for response in bad:
            with self.subTest(response=response), self.assertRaises(ModelError):
                parse_message(response)

    def test_parser_drops_private_reasoning(self):
        message = final("简短答案") | {"reasoning_content": "private", "other": "ignored"}
        parsed = parse_message({"choices": [{"message": message, "finish_reason": "stop"}]})
        self.assertEqual(parsed, final("简短答案"))

    def test_duplicate_ids_in_single_message_rejected(self):
        decision = call("search", {"query": "x"})
        decision["tool_calls"] *= 2
        with self.assertRaises(ModelError):
            parse_message({"choices": [{"message": decision, "finish_reason": "tool_calls"}]})

    def test_real_adapter_http_roundtrip_without_external_api(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                message = call("calculator", {"expression": "6*7"}) if len(received) == 1 else final("42")
                payload = json.dumps({"choices": [{"message": message, "finish_reason": "tool_calls" if len(received) == 1 else "stop"}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "LLM_API_KEY": "test-only-placeholder", "LLM_MODEL": "test-model",
                "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            }):
                result = Runtime(Store(Path(folder) / "test.db"), ChatModel(), make_registry()).chat("a", "1", "计算 6*7")
                self.assertEqual(result["answer"], "42")
                self.assertEqual(received[0]["tool_choice"], "auto")
                self.assertEqual(received[1]["messages"][-1]["tool_call_id"], "call_1")
                self.assertEqual(json.loads(received[1]["messages"][-1]["content"])["data"]["value"], 42)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_insecure_remote_endpoint_rejected(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_MODEL": "test", "LLM_BASE_URL": "http://example.com/v1"}):
            with self.assertRaises(ModelError):
                ChatModel()

    def test_http_errors_and_redirects_do_not_leak_body(self):
        from urllib.error import HTTPError
        from io import BytesIO
        from agent.llm import NoRedirect
        with patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_MODEL": "test", "LLM_BASE_URL": "https://example.com/v1"}):
            for status in [401, 429, 500, 302]:
                with self.subTest(status=status), patch("agent.llm.build_opener") as opener:
                    opener.return_value.open.side_effect = HTTPError("https://example.com", status, "failure", {}, BytesIO(b"private-secret"))
                    with self.assertRaises(ModelError) as error:
                        ChatModel().complete([], [])
                    self.assertIn(str(status), str(error.exception))
                    self.assertNotIn("private-secret", str(error.exception))
            self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))

    def test_network_timeout_is_reported(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_MODEL": "test", "LLM_BASE_URL": "https://example.com/v1"}), patch("agent.llm.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError()
            with self.assertRaisesRegex(ModelError, "超时"):
                ChatModel().complete([], [])


if __name__ == "__main__":
    unittest.main()
