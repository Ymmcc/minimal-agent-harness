"""Opt-in live smoke check. Reads credentials only from process environment."""
import argparse
import getpass
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.llm import ChatModel, ModelError
from agent.runtime import Runtime
from agent.store import Store
from agent.tools import make_registry


class CheckFailed(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="真实模型小规模验收；密钥仅从环境或本机遮罩输入读取")
    parser.add_argument("--prompt", action="store_true", help="在交互终端遮罩输入密钥")
    parser.add_argument("--base-url", help="Chat Completions 基础地址")
    parser.add_argument("--model", help="模型 ID")
    args = parser.parse_args()
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    report_path = Path("data/live_check.json")
    report_path.parent.mkdir(exist_ok=True)
    report = {"status": "running", "date": datetime.now(timezone.utc).isoformat(),
              "model": os.environ.get("LLM_MODEL", ""), "checks": []}
    model = None
    started = time.monotonic()

    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    def require(condition, description):
        if not condition:
            raise CheckFailed(description)

    try:
        save()
        if args.prompt:
            if not sys.stdin.isatty():
                raise ModelError("遮罩输入需要本机交互终端；请在终端中启动。")
            os.environ["LLM_API_KEY"] = getpass.getpass("API Key (hidden): ").strip()
        model = ChatModel()
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "live.db")
            registry = make_registry()
            runtime = Runtime(store, model, registry, max_steps=4)

            def ask(label, sid, text):
                before = len(store.history("live-user", sid))
                result = runtime.chat("live-user", sid, text)
                events = [e for e in store.traces("live-user", sid) if e["run"] == result["run_id"]]
                if any(e["event"] == "error" for e in events):
                    # Runtime already sanitizes model/network errors.
                    raise CheckFailed(result["answer"])
                if "步数上限" in result["answer"]:
                    raise CheckFailed("本轮模型未在四步内完成")
                rows = store.history("live-user", sid)[before:]
                tools = [(m.get("name"), json.loads(m["content"])) for _, m in rows if m["role"] == "tool"]
                report["checks"].append({"name": label, "passed": False,
                                         "tools": [name for name, _ in tools]})
                save()
                return result["answer"], tools

            def passed():
                report["checks"][-1]["passed"] = True
                save()
                print("PASS " + report["checks"][-1]["name"], flush=True)

            answer, tools = ask("直接回复", "main", "你好，请只回复一句简短问候，不需要使用工具。")
            require(bool(answer.strip()) and not tools, "直接回复未按预期完成")
            passed()

            answer, tools = ask("记录对话信息", "main", "我叫小林，先记住这个名字即可，不要创建待办。")
            require(not tools, "记录名字时产生了不必要的工具调用")
            passed()
            answer, tools = ask("纯对话追问", "main", "我叫什么名字？")
            require("小林" in answer and not tools, "纯对话追问未正确使用历史")
            passed()

            answer, tools = ask("计算后新增待办", "main",
                                "请先用 calculator 计算 18*7，再用 todo 新增一条待办，内容为复习计算结果并包含该数值。")
            require(any(n == "calculator" and r.get("ok") and r["data"].get("value") == 126 for n, r in tools),
                    "未得到 calculator 的正确结果")
            require(any(n == "todo" and r.get("ok") for n, r in tools), "未成功调用 todo")
            with store.connect() as db:
                item = db.execute("SELECT id,text FROM todos WHERE user=? AND sid=?", ("live-user", "main")).fetchone()
                require(item is not None and "126" in item["text"], "待办未保存正确计算值")
                item_id = item["id"]
            passed()

            answer, tools = ask("带工具的追问", "main", "把刚才新增的那条待办标为完成。")
            with store.connect() as db:
                done = db.execute("SELECT done FROM todos WHERE id=?", (item_id,)).fetchone()["done"]
            require(done == 1 and any(n == "todo" for n, _ in tools), "追问未更新刚才的待办")
            passed()

            answer, tools = ask("独立 session", "other", "请使用 todo 查询这个会话的待办列表。")
            require(any(n == "todo" and r.get("ok") and r["data"].get("items") == [] for n, r in tools),
                    "独立 session 未返回空待办列表")
            passed()

            runtime = Runtime(Store(store.path), model, registry, max_steps=4)
            answer, tools = ask("重建 Runtime 后恢复", "main", "请使用 todo 查询当前会话的所有待办。")
            require(any(n == "todo" and r.get("ok") and any(i["id"] == item_id and i["done"] == 1
                        for i in r["data"].get("items", [])) for n, r in tools), "恢复后待办状态不正确")
            passed()

            answer, tools = ask("模拟搜索", "main", "请使用 search 查 Agent runtime，并说明这个工具是不是模拟搜索。")
            require(any(n == "search" and r.get("ok") and r["data"].get("mock") for n, r in tools),
                    "未使用模拟搜索工具")
            require(any(word in answer.lower() for word in ("模拟", "演示", "mock")), "未说明搜索是模拟数据")
            passed()
        report["status"] = "passed"
    except (CheckFailed, ModelError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    except Exception:
        report["status"] = "failed"
        report["error"] = "验证进程出现异常；未输出内部异常信息。"
    finally:
        os.environ.pop("LLM_API_KEY", None)
        if model is not None:
            model.key = ""
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save()
    print("Live check: " + report["status"], flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
