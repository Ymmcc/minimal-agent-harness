"""SQLite is the source of truth; every operation uses a short transaction."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class BusyError(Exception):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    user TEXT, sid TEXT, owner TEXT, lease REAL DEFAULT 0,
                    summary TEXT DEFAULT '', summary_until INTEGER DEFAULT 0,
                    PRIMARY KEY(user, sid)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, user TEXT, sid TEXT, body TEXT
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(user, sid, id);
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY, user TEXT, sid TEXT,
                    text TEXT, done INTEGER DEFAULT 0, call_id TEXT,
                    UNIQUE(user, sid, call_id)
                );
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY, user TEXT, sid TEXT, run TEXT,
                    event TEXT, detail TEXT, created REAL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def claim(self, user, sid):
        owner = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO sessions(user,sid) VALUES (?,?)", (user, sid))
            changed = db.execute(
                "UPDATE sessions SET owner=?,lease=? WHERE user=? AND sid=? AND lease<=?",
                (owner, time.time() + 300, user, sid, time.time()),
            ).rowcount
            if not changed:
                raise BusyError("该 session 正在执行，请稍后重试或切换 session。")
        return owner

    @staticmethod
    def check_owner(db, user, sid, owner):
        row = db.execute("SELECT owner,lease FROM sessions WHERE user=? AND sid=?", (user, sid)).fetchone()
        if not row or row["owner"] != owner or row["lease"] <= time.time():
            raise BusyError("会话租约已过期；当前执行停止，请重新发起请求。")

    def release(self, user, sid, owner):
        with self.connect() as db:
            db.execute("UPDATE sessions SET owner=NULL,lease=0 WHERE user=? AND sid=? AND owner=?",
                       (user, sid, owner))

    def append(self, user, sid, owner, message):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.check_owner(db, user, sid, owner)
            db.execute("INSERT INTO messages(user,sid,body) VALUES (?,?,?)",
                       (user, sid, json.dumps(message, ensure_ascii=False)))

    def history(self, user, sid):
        with self.connect() as db:
            rows = db.execute("SELECT id,body FROM messages WHERE user=? AND sid=? ORDER BY id", (user, sid))
            return [(row["id"], json.loads(row["body"])) for row in rows]

    def memory(self, user, sid):
        with self.connect() as db:
            row = db.execute("SELECT summary,summary_until FROM sessions WHERE user=? AND sid=?", (user, sid)).fetchone()
            return (row["summary"], row["summary_until"]) if row else ("", 0)

    def save_memory(self, user, sid, owner, summary, until):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.check_owner(db, user, sid, owner)
            db.execute("UPDATE sessions SET summary=?,summary_until=? WHERE user=? AND sid=?",
                       (summary, until, user, sid))

    def todo(self, user, sid, owner, call_id, action, text=None, item_id=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.check_owner(db, user, sid, owner)
            if action == "add":
                db.execute("INSERT OR IGNORE INTO todos(user,sid,text,call_id) VALUES (?,?,?,?)",
                           (user, sid, text, call_id))
            elif action == "done":
                count = db.execute("UPDATE todos SET done=1 WHERE user=? AND sid=? AND id=?",
                                   (user, sid, item_id)).rowcount
                if not count:
                    raise ValueError("待办不存在于当前 session")
            rows = db.execute("SELECT id,text,done FROM todos WHERE user=? AND sid=? ORDER BY id", (user, sid))
            return {"items": [dict(row) for row in rows]}

    def trace(self, user, sid, run, event, detail):
        # No HTTP headers, API keys or private model reasoning are recorded.
        with self.connect() as db:
            db.execute("INSERT INTO traces(user,sid,run,event,detail,created) VALUES (?,?,?,?,?,?)",
                       (user, sid, run, event, json.dumps(detail, ensure_ascii=False), time.time()))

    def traces(self, user, sid):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT run,event,detail,created FROM traces WHERE user=? AND sid=? ORDER BY id", (user, sid))]
