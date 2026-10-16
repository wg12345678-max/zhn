"""
审计日志模块：记录学生操作全过程，支持回溯与导出
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


class AuditLogger:
    """
    审计日志记录器

    用法:
        logger = AuditLogger("session_001")
        logger.log_step(1, "选择企业类型", {"type": "能源装备"})
        logger.log_rule_hit({"rule_name": "明文身份证号", "deduction": 8})
        df = logger.get_audit_trail()
    """

    def __init__(self, session_id: str, log_dir: str = "./audit_logs"):
        self.session_id = session_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{session_id}.jsonl")
        self._events: List[Dict] = []

    def log_step(
        self,
        step_no: int,
        action: str,
        params: Dict[str, Any] = None,
        result: Dict[str, Any] = None,
        duration_seconds: float = None,
    ):
        event = {
            "type": "step",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step_no": step_no,
            "action": action,
            "params": params or {},
            "result": result or {},
            "duration_seconds": duration_seconds,
        }
        self._write(event)
        self._events.append(event)

    def log_rule_hit(self, rule_hit: Dict):
        event = {
            "type": "rule_hit",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            **rule_hit,
        }
        self._write(event)
        self._events.append(event)

    def log_decision(self, decision_name: str, value: Any, reasoning: str = None):
        event = {
            "type": "decision",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "decision_name": decision_name,
            "value": value,
            "reasoning": reasoning,
        }
        self._write(event)
        self._events.append(event)

    def log_random_event(self, event_name: str, student_response: str, outcome: str):
        event = {
            "type": "random_event",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_name": event_name,
            "student_response": student_response,
            "outcome": outcome,
        }
        self._write(event)
        self._events.append(event)

    def log_error(self, error_msg: str, context: Dict = None):
        event = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "error_msg": error_msg,
            "context": context or {},
        }
        self._write(event)

    def _write(self, event: Dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def get_audit_trail(self) -> pd.DataFrame:
        events = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        return pd.DataFrame(events)

    def get_step_summary(self) -> pd.DataFrame:
        df = self.get_audit_trail()
        if not df.empty and "type" in df.columns:
            steps = df[df["type"] == "step"].copy()
            if not steps.empty:
                steps["step_no"] = steps["step_no"].astype(int)
                return steps.sort_values("step_no")
        return pd.DataFrame()

    def export_csv(self, path: str = None) -> str:
        if path is None:
            path = os.path.join(self.log_dir, f"{self.session_id}_audit.csv")
        df = self.get_audit_trail()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def clear(self):
        self._events = []
        if os.path.exists(self.log_path):
            os.remove(self.log_path)


_loggers: Dict[str, AuditLogger] = {}


def get_logger(session_id: str) -> AuditLogger:
    if session_id not in _loggers:
        _loggers[session_id] = AuditLogger(session_id)
    return _loggers[session_id]


if __name__ == "__main__":
    logger = AuditLogger("test_session_001")
    logger.log_step(1, "选择企业类型", {"type": "能源装备"})
    logger.log_step(2, "运行模块一", {"清洗强度": 0.6}, {"异常工单数": 12})
    logger.log_rule_hit({
        "rule_name": "明文身份证号",
        "law_article": "个保法第13条",
        "severity": "严重",
        "deduction": 8,
        "description": "记录 REC-2026-000042 含明文身份证号",
    })
    logger.log_decision("脱敏策略", "掩码", "选择掩码以平衡合规与可用性")
    logger.log_random_event("数据泄露", "立即召回", "合规度-20，成本+30%")
    df = logger.get_audit_trail()
    print(f"共 {len(df)} 条审计记录")
    print(f"事件类型分布:\n{df['type'].value_counts()}")
    path = logger.export_csv()
    print(f"已导出至: {path}")
