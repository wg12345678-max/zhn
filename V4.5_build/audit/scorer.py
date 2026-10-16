"""
20 步逐项赋分系统 + 四维能力画像
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ScoringRule:
    step_no: int
    max_score: float
    deduct_on: List[str]
    deduct_amount: float = 2.0
    min_score: float = 0.0


STEP_SCORING_RULES: Dict[int, ScoringRule] = {
    1: ScoringRule(1, 2.0, ["未选择企业类型"]),
    2: ScoringRule(2, 6.0, ["答错"], deduct_amount=2.0),
    3: ScoringRule(3, 2.0, ["跳过不看"]),
    4: ScoringRule(4, 4.0, ["漏选关键来源"], deduct_amount=2.0),
    5: ScoringRule(5, 2.0, ["直接运行不查看配置"]),
    6: ScoringRule(6, 10.0, ["误判脏数据类型", "忽略敏感字段"], deduct_amount=2.5),
    7: ScoringRule(7, 4.0, ["调到极端值"]),
    8: ScoringRule(8, 6.0, ["选择不处理"], deduct_amount=3.0),
    9: ScoringRule(9, 6.0, ["分类错误"], deduct_amount=2.0),
    10: ScoringRule(10, 2.0, ["直接运行不查看参数"]),
    11: ScoringRule(11, 6.0, ["未调整直接运行"]),
    12: ScoringRule(12, 4.0, ["不做任何调整就确认"]),
    13: ScoringRule(13, 12.0, ["错误应对"], deduct_amount=4.0),
    14: ScoringRule(14, 4.0, ["选择什么都不做"]),
    15: ScoringRule(15, 2.0, ["直接运行不查看参数"]),
    16: ScoringRule(16, 2.0, ["跳过不看"]),
    17: ScoringRule(17, 8.0, ["判断错误"], deduct_amount=2.0),
    18: ScoringRule(18, 6.0, ["判断错误"], deduct_amount=2.0),
    19: ScoringRule(19, 4.0, ["备忘录少于100字", "AI学伴反问未回复"], deduct_amount=2.0),
    20: ScoringRule(20, 2.0, ["未导出"]),
}


@dataclass
class StepScore:
    step_no: int
    action: str
    raw_score: float
    deductions: List[tuple]
    final_score: float
    max_score: float


@dataclass
class SessionScore:
    session_id: str
    step_scores: List[StepScore]
    total_raw: float
    total_deductions: float
    total_final: float
    total_max: float
    percentage: float


@dataclass
class Abilities:
    data_quality: float
    compliance: float
    asset_operation: float
    emergency_response: float

    def to_dict(self) -> dict:
        return {
            "data_quality": round(self.data_quality, 1),
            "compliance": round(self.compliance, 1),
            "asset_operation": round(self.asset_operation, 1),
            "emergency_response": round(self.emergency_response, 1),
        }

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 95: return "S"
        if score >= 85: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        return "D"

    def grades(self) -> dict:
        return {
            "data_quality": self._grade(self.data_quality),
            "compliance": self._grade(self.compliance),
            "asset_operation": self._grade(self.asset_operation),
            "emergency_response": self._grade(self.emergency_response),
        }


def score_session(audit_df: pd.DataFrame, session_id: str = None) -> SessionScore:
    if session_id is None and not audit_df.empty:
        session_id = audit_df.iloc[0].get("session_id", "unknown")

    step_scores = []
    total_raw = 0.0
    total_deductions = 0.0
    total_max = 0.0

    steps_df = audit_df[audit_df["type"] == "step"].copy() if "type" in audit_df.columns else pd.DataFrame()

    for step_no in sorted(STEP_SCORING_RULES.keys()):
        rule = STEP_SCORING_RULES[step_no]
        step_events = steps_df[steps_df["step_no"] == step_no] if not steps_df.empty else pd.DataFrame()

        raw_score = rule.max_score
        deductions = []

        if not step_events.empty:
            for _, event in step_events.iterrows():
                action_desc = str(event.get("action", ""))
                params = event.get("params", {})
                result = event.get("result", {})
                for keyword in rule.deduct_on:
                    if (keyword in action_desc or keyword in str(params) or keyword in str(result)):
                        deductions.append((keyword, rule.deduct_amount))
                        raw_score -= rule.deduct_amount
        else:
            deductions.append(("未执行", rule.max_score))
            raw_score = 0.0

        final_score = max(raw_score, rule.min_score)
        total_raw += rule.max_score
        total_deductions += rule.max_score - final_score
        total_max += rule.max_score

        step_scores.append(StepScore(
            step_no=step_no,
            action=step_events.iloc[0].get("action", "未执行") if not step_events.empty else "未执行",
            raw_score=rule.max_score,
            deductions=deductions,
            final_score=final_score,
            max_score=rule.max_score,
        ))

    total_final = sum(s.final_score for s in step_scores)
    percentage = (total_final / total_max * 100) if total_max > 0 else 0.0

    return SessionScore(
        session_id=session_id or "unknown",
        step_scores=step_scores,
        total_raw=total_raw,
        total_deductions=total_deductions,
        total_final=total_final,
        total_max=total_max,
        percentage=percentage,
    )


def compute_abilities(score: SessionScore, audit_df: pd.DataFrame = None) -> Abilities:
    step_map = {s.step_no: s for s in score.step_scores}

    dq_steps = [6, 7, 8, 9]
    comp_steps = [4, 12, 13, 14, 17, 18]
    ao_steps = [11, 15, 16, 17]
    er_steps = [13, 14]

    def _calc(steps):
        total = sum(step_map[s].max_score for s in steps if s in step_map)
        earned = sum(step_map[s].final_score for s in steps if s in step_map)
        return (earned / total * 100) if total > 0 else 0.0

    return Abilities(
        data_quality=_calc(dq_steps),
        compliance=_calc(comp_steps),
        asset_operation=_calc(ao_steps),
        emergency_response=_calc(er_steps),
    )


if __name__ == "__main__":
    from audit.audit_logger import AuditLogger

    logger = AuditLogger("score_test")
    logger.log_step(1, "选择企业类型", {"type": "能源装备"})
    logger.log_step(6, "异常工单处理", {"已处理": 5})
    logger.log_step(13, "应对数据泄露", {"响应": "召回+整改"})

    df = logger.get_audit_trail()
    score = score_session(df, "score_test")
    print(f"总分: {score.total_final}/{score.total_max} ({score.percentage:.1f}%)")

    abilities = compute_abilities(score, df)
    print(f"能力画像: {abilities.to_dict()}")
    print(f"等级: {abilities.grades()}")
