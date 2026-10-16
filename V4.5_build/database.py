"""
数据库模块：SQLite/PostgreSQL 自动切换 + 200 个测试账号生成
- 默认 SQLite（零配置，下载即用）
- 检测到 DATABASE_URL 环境变量时自动切换 PostgreSQL
- 200 个测试账号在初始化时烧入数据库，无个人信息
"""
import os
import sqlite3
import hashlib
import secrets
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StudentAccount:
    username: str
    password: str
    display_name: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    username    TEXT PRIMARY KEY,
    password    TEXT NOT NULL,
    display_name TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_login  TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    username     TEXT NOT NULL REFERENCES students(username),
    group_name   TEXT,
    semester     INTEGER,
    total_score  REAL DEFAULT 0,
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ability_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    data_quality REAL, compliance REAL, asset_operation REAL, emergency_response REAL,
    total_score  REAL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id);
"""


def get_connection():
    """根据环境变量自动选择数据库，SQLite 端启用 WAL + 超时 + 重试，规避 disk I/O error"""
    url = os.getenv("DATABASE_URL")
    if url and url.startswith("postgresql"):
        try:
            import psycopg2
            return psycopg2.connect(url), "postgresql"
        except ImportError:
            pass
    # 默认 SQLite：使用绝对路径 + 超时 + WAL 模式
    # 若默认路径所在文件系统不支持 SQLite（如部分网络/虚拟挂载），
    # 自动回退到系统临时目录，保证数据库可用
    preferred = os.getenv("SQLITE_PATH", "./sim.db")
    candidates = [os.path.abspath(preferred)]
    import tempfile as _tempfile
    candidates.append(os.path.join(_tempfile.gettempdir(), "sim.db"))

    last_err = None
    for db_path in candidates:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        try:
            # 先用轻量写操作探测该路径是否支持 SQLite
            _probe = sqlite3.connect(db_path, timeout=10)
            _probe.execute("CREATE TABLE IF NOT EXISTS _probe(x)")
            _probe.execute("DROP TABLE IF EXISTS _probe")
            _probe.commit()
            _probe.close()
        except sqlite3.OperationalError as e:
            last_err = e
            continue

        for attempt in range(3):
            try:
                conn = sqlite3.connect(db_path, timeout=30)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                except Exception:
                    pass
                return conn, "sqlite"
            except sqlite3.OperationalError as e:
                last_err = e
                import time
                time.sleep(0.5)
    raise last_err


def init_db():
    """初始化数据库表结构"""
    conn, kind = get_connection()
    try:
        cur = conn.cursor()
        cur.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return kind


def hash_password(password: str) -> str:
    """SHA-256 + salt 存储，不存明文"""
    salt = "ai_dg_sim_2026"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def verify_login(username: str, password: str) -> Optional[str]:
    """验证登录，返回 display_name 或 None"""
    conn, _ = get_connection()
    try:
        cur = conn.cursor()
        if isinstance(cur, sqlite3.Cursor):
            cur = conn.cursor()
        cur.execute("SELECT display_name FROM students WHERE username=? AND password=?", (username, hash_password(password)))
        row = cur.fetchone()
        if row:
            return row[0]
        return None
    finally:
        conn.close()


def create_students(accounts: List[StudentAccount]):
    """批量写入学生账号"""
    conn, _ = get_connection()
    try:
        cur = conn.cursor()
        for acc in accounts:
            cur.execute(
                "INSERT OR IGNORE INTO students (username, password, display_name) VALUES (?, ?, ?)",
                (acc.username, hash_password(acc.password), acc.display_name),
            )
        conn.commit()
    finally:
        conn.close()


def save_session(session_id: str, username: str, group_name: str, semester: int):
    conn, _ = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO sessions (session_id, username, group_name, semester) VALUES (?, ?, ?, ?)",
            (session_id, username, group_name, semester),
        )
        conn.commit()
    finally:
        conn.close()


def finish_session(session_id: str, total_score: float, abilities: Dict[str, float]):
    conn, _ = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE sessions SET finished_at=datetime('now'), total_score=? WHERE session_id=?",
                    (total_score, session_id))
        cur.execute(
            "INSERT INTO ability_records (session_id, data_quality, compliance, asset_operation, emergency_response, total_score) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, abilities.get("data_quality", 0), abilities.get("compliance", 0),
             abilities.get("asset_operation", 0), abilities.get("emergency_response", 0), total_score),
        )
        conn.commit()
    finally:
        conn.close()


def generate_200_accounts(prefix: str = "dg", seed: int = 20261015) -> List[StudentAccount]:
    """
    生成 200 个学生账号（用户名 dg001~dg200，密码为可记忆的强密码）

    说明:
        - 无任何真实个人信息（姓名、学号、身份证等），仅匿名编号
        - 密码格式: 前缀+编号+随机字符，兼顾强度与可发放
    """
    import random
    import string
    random.seed(seed)
    chars = string.ascii_letters + string.digits
    accounts = []
    for i in range(1, 201):
        username = f"{prefix}{i:03d}"
        # 8位密码: 前缀后3位 + 5位随机字母数字
        password = f"{i:03d}{''.join(random.choices(chars, k=5))}"
        accounts.append(StudentAccount(
            username=username,
            password=password,
            display_name=f"学生{username}",
        ))
    return accounts


if __name__ == "__main__":
    init_db()
    accounts = generate_200_accounts()
    create_students(accounts)
    print(f"已生成并写入 {len(accounts)} 个账号")
    print(f"示例: {accounts[0].username} / {accounts[0].password}")
    print(f"登录验证测试: {verify_login(accounts[0].username, accounts[0].password)}")
    print(f"错误密码测试: {verify_login(accounts[0].username, 'wrong')}")
