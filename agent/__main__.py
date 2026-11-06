"""CLI entry point. Each terminal can select its own persistent session."""
import argparse
import json

from .llm import ChatModel, MockModel, ModelError
from .runtime import Runtime
from .store import Store, BusyError
from .tools import make_registry


def main():
    parser = argparse.ArgumentParser(description="最小 Agent：默认离线演示，--live 使用真实模型")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--user", default="demo")
    parser.add_argument("--session", default=None, help="不填时生成新会话；恢复时传入原 ID")
    parser.add_argument("--db", default="data/agent.db")
    parser.add_argument("--message", help="执行单条消息后退出")
    args = parser.parse_args()
    import uuid
    sid = args.session or uuid.uuid4().hex[:12]
    store = Store(args.db)
    try:
        runtime = Runtime(store, ChatModel() if args.live else MockModel(), make_registry())
    except ModelError as exc:
        parser.exit(2, str(exc) + "\n")
    print(f"模式={'真实模型' if args.live else '离线 mock'} user={args.user} session={sid}")
    if not args.message:
        print("命令：/session ID 切换会话，/trace 查看日志，/quit 退出。")
        print("演示：计算 12*(3+4) | 再乘以 2 | 记待办 周五复习 | 待办列表 | 搜索 Agent")
    while True:
        try:
            text = args.message if args.message is not None else input(f"[{sid}] 你> ")
            if text == "/quit":
                break
            if text.startswith("/session "):
                sid = text.split(maxsplit=1)[1].strip()
                print("已切换 session=" + sid)
            elif text == "/trace":
                print(json.dumps(store.traces(args.user, sid), ensure_ascii=False, indent=2))
            else:
                result = runtime.chat(args.user, sid, text)
                print("Agent> " + result["answer"] + "\nrun_id=" + result["run_id"])
        except (BusyError, ValueError) as exc:
            print(str(exc))
            if args.message is not None:
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\n已退出；会话已保存。")
            break
        if args.message is not None:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
