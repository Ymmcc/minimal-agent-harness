"""The agent loop: load context -> model -> tools -> repeat / final answer."""
import json
import time
import uuid

from .context import ContextBuilder, ContextOverflow
from .llm import ModelError, parse_message
from .tools import ToolContext


class Runtime:
    def __init__(self, store, model, registry, max_steps=6, context_budget=18000):
        if not 1 <= max_steps <= 8:
            raise ValueError("max_steps 必须在 1..8 之间")
        self.store, self.model, self.registry = store, model, registry
        self.max_steps = max_steps
        self.context = ContextBuilder(store, context_budget)

    def _repair_interrupted(self, user, sid, owner):
        """Close dangling calls after a process crash, without replaying side effects."""
        history = self.store.history(user, sid)
        answered = {msg["tool_call_id"] for _, msg in history if msg["role"] == "tool"}
        for _, msg in history:
            for call in msg.get("tool_calls", []):
                if call["id"] not in answered:
                    self.store.append(user, sid, owner, {
                        "role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                        "content": json.dumps({"ok": False, "error": "interrupted", "message": "上次执行中断，结果未知；写操作不得自动重放，请先查询状态。"}, ensure_ascii=False),
                    })
        if history and history[-1][1]["role"] != "assistant":
            self.store.append(user, sid, owner, {"role": "assistant", "content": "上次执行中断，请查询工具状态后继续。"})
        elif history and history[-1][1].get("tool_calls"):
            self.store.append(user, sid, owner, {"role": "assistant", "content": "上次执行中断，请查询工具状态后继续。"})

    def chat(self, user, sid, text):
        if not text.strip() or len(text) > 4000:
            raise ValueError("输入不能为空，且最多 4000 字符。")
        if not user.strip() or not sid.strip() or len(user) > 100 or len(sid) > 100:
            raise ValueError("user/session 必须是 1..100 字符。")
        owner = self.store.claim(user, sid)
        run = uuid.uuid4().hex
        started = time.monotonic()

        def append(message):
            self.store.append(user, sid, owner, message)

        def finish(answer):
            append({"role": "assistant", "content": answer})
            self.store.trace(user, sid, run, "finish", {"elapsed_ms": round((time.monotonic() - started) * 1000)})
            return {"answer": answer, "run_id": run}

        try:
            self._repair_interrupted(user, sid, owner)
            append({"role": "user", "content": text})
            self.store.trace(user, sid, run, "start", {"input_chars": len(text)})
            schemas = self.registry.schemas()
            used_ids = {c["id"] for _, msg in self.store.history(user, sid) for c in msg.get("tool_calls", [])}
            for step in range(self.max_steps):
                messages = self.context.build(user, sid, owner, schemas)
                raw = self.model.complete(messages, schemas)
                # Validate mock/test adapters too; all adapters have the same contract.
                decision = parse_message({"choices": [{"message": raw, "finish_reason": "tool_calls" if raw.get("tool_calls") else "stop"}]})
                calls = decision.get("tool_calls", [])
                if any(call["id"] in used_ids for call in calls):
                    raise ModelError("模型重复使用历史工具调用 id，本次执行停止。")
                if not calls:
                    return finish(decision["content"])
                append(decision)
                self.store.trace(user, sid, run, "decision", {"step": step + 1, "tools": [c["function"]["name"] for c in calls]})
                for call in calls:
                    used_ids.add(call["id"])
                    name = call["function"]["name"]
                    tool_started = time.monotonic()
                    try:
                        arguments = json.loads(call["function"]["arguments"])
                        result = self.registry.execute(name, arguments, ToolContext(self.store, user, sid, owner, call["id"]))
                    except (ValueError, RecursionError):
                        result = {"ok": False, "error": "invalid_json", "message": "工具参数必须是有效 JSON。"}
                    content = json.dumps(result, ensure_ascii=False)
                    if len(content) > 2500:
                        content = json.dumps({"ok": result["ok"], "truncated": True, "preview": content[:2000]}, ensure_ascii=False)
                    append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": content})
                    self.store.trace(user, sid, run, "tool", {"call_id": call["id"], "name": name, "ok": result["ok"],
                                                              "elapsed_ms": round((time.monotonic() - tool_started) * 1000)})
            return finish("已达到本轮执行步数上限；部分工具可能已经执行，请查看待办或 trace 后继续。")
        except (ModelError, ContextOverflow) as exc:
            self.store.trace(user, sid, run, "error", {"type": type(exc).__name__})
            return finish(str(exc))
        finally:
            self.store.release(user, sid, owner)
