"""Native Chat Completions tool protocol; urllib handles HTTP, not orchestration."""
import json
import os
import re
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener


class ModelError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward Authorization to a redirected host.
        return None


def parse_message(response):
    """Normalize only public content/tool calls. Never persist reasoning_content."""
    try:
        choice = response["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict):
            raise ValueError("message 必须是对象")
        if choice["finish_reason"] not in ("stop", "tool_calls"):
            raise ValueError("模型输出未正常结束")
        content = message.get("content")
        calls = message.get("tool_calls") or []
        if content is not None and (not isinstance(content, str) or len(content) > 8000):
            raise ValueError("无效文本")
        if not isinstance(calls, list) or len(calls) > 4:
            raise ValueError("每步最多四个工具调用")
        if not calls and not (content and content.strip()):
            raise ValueError("空响应")
        result = {"role": "assistant", "content": content}
        if calls:
            cleaned, seen = [], set()
            for call in calls:
                call_id, function = call["id"], call["function"]
                if not isinstance(call_id, str) or not call_id or len(call_id) > 200 or call_id in seen:
                    raise ValueError("无效或重复调用 id")
                seen.add(call_id)
                name, arguments = function["name"], function["arguments"]
                if call.get("type") != "function" or not isinstance(name, str) or len(name) > 100:
                    raise ValueError("无效工具名称或类型")
                if not isinstance(arguments, str) or len(arguments) > 4000:
                    raise ValueError("无效工具参数")
                # Invalid argument JSON is a recoverable TOOL error, not a lost turn.
                cleaned.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
            result["tool_calls"] = cleaned
        return result
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelError("模型响应格式无效、为空或被截断") from exc


class ChatModel:
    def __init__(self):
        self.key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        url = urlparse(self.base)
        local = url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
        if not self.key or not self.model or not url.hostname or not (url.scheme == "https" or local):
            raise ModelError("请设置 LLM_API_KEY、LLM_MODEL、LLM_BASE_URL（HTTPS，或本地 HTTP）。")
        if url.username or url.password or url.query or url.fragment:
            raise ModelError("LLM_BASE_URL 不得含账号、密码、查询参数或片段")

    def complete(self, messages, tools):
        wire_messages = [{k: v for k, v in msg.items() if k != "name"} for msg in messages]
        body = {"model": self.model, "messages": wire_messages, "tools": tools,
                "tool_choice": "auto", "max_tokens": 1200}
        request = Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with build_opener(NoRedirect()).open(request, timeout=20) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ModelError("模型响应过大")
            return parse_message(json.loads(raw))
        except HTTPError as exc:
            # Response bodies may echo credentials: report status only.
            raise ModelError(f"模型 HTTP 请求失败（状态 {exc.code}），请检查配置或配额。") from None
        except (URLError, TimeoutError, OSError):
            raise ModelError("模型网络请求失败或超时，请稍后重试。") from None
        except (ValueError, UnicodeError, RecursionError):
            raise ModelError("模型返回的内容不是有效 JSON。") from None


class MockModel:
    """Explicitly a small deterministic demo, NOT an LLM or the real decision path."""
    @staticmethod
    def call(name, arguments):
        return {"role": "assistant", "content": "执行工具以获取结果。", "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]}

    def complete(self, messages, tools):
        last = messages[-1]
        if last["role"] == "tool":
            return {"role": "assistant", "content": "工具结果：" + last["content"]}
        text = last["content"].strip()
        if text.startswith("计算 "):
            return self.call("calculator", {"expression": text[3:]})
        if text.startswith("再乘以 "):
            for msg in reversed(messages):
                if msg["role"] == "tool" and msg.get("name") == "calculator":
                    result = json.loads(msg["content"])
                    if result.get("ok"):
                        return self.call("calculator", {"expression": f"({result['data']['value']})*({text[4:]})"})
            return {"role": "assistant", "content": "请先计算一个结果。"}
        if text.startswith("搜索 "):
            return self.call("search", {"query": text[3:]})
        if text.startswith("记待办 "):
            return self.call("todo", {"action": "add", "text": text[4:]})
        if text == "待办列表":
            return self.call("todo", {"action": "list"})
        if re.fullmatch(r"完成待办 \d+", text):
            return self.call("todo", {"action": "done", "item_id": int(text.split()[-1])})
        if text == "我叫什么":
            for msg in reversed(messages[:-1]):
                if msg["role"] == "user":
                    match = re.search(r"我叫([^\n，。]+)", msg["content"])
                    if match:
                        return {"role": "assistant", "content": "你叫" + match[1] + "。"}
            return {"role": "assistant", "content": "当前上下文里没有你的名字。"}
        return {"role": "assistant", "content": "[离线演示] 已收到：" + text}
