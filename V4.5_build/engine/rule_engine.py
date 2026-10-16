"""
确定性规则引擎
每条规则命中输出: 规则名/法条依据/扣分/严重级别/建议操作
"""
from dataclasses import dataclass, field
from typing import List, Optional
import re
import pandas as pd


@dataclass
class RuleHit:
    rule_name: str
    law_article: str
    severity: str
    deduction: float
    description: str
    suggested_action: str
    record_id: Optional[str] = None
    field_name: Optional[str] = None


@dataclass
class RuleSet:
    name: str
    rules: List[dict] = field(default_factory=list)


PII_RULES = RuleSet(name="个人信息保护规则", rules=[
    {"name": "明文身份证号", "pattern": r"\d{18}[Xx]?|\d{17}[Xx]", "law": "个保法第13条", "severity": "严重", "deduction": 8, "desc": "明文身份证号未脱敏", "action": "掩码/泛化/加密/差分隐私"},
    {"name": "明文手机号", "pattern": r"1[3-9]\d{9}", "law": "个保法第13条", "severity": "高", "deduction": 6, "desc": "明文手机号未脱敏", "action": "掩码/泛化/加密"},
    {"name": "明文银行卡号", "pattern": r"\d{16,19}", "law": "个保法第28条", "severity": "严重", "deduction": 8, "desc": "明文银行卡号未脱敏", "action": "掩码/加密/差分隐私"},
])

DATA_QUALITY_RULES = RuleSet(name="数据质量规则", rules=[
    {"name": "缺失值过多", "threshold": 0.3, "law": "数据质量管理规范第4条", "severity": "中", "deduction": 3, "desc": "字段缺失率超过30%", "action": "填充/删除/重新采集"},
    {"name": "重复记录", "threshold": 1, "law": "数据质量管理规范第5条", "severity": "低", "deduction": 1, "desc": "存在完全重复记录", "action": "去重"},
    {"name": "异常值偏离", "threshold": 3.0, "law": "数据质量管理规范第6条", "severity": "中", "deduction": 3, "desc": "数值字段Z-score超过3", "action": "截尾/修正/剔除"},
])

COPYRIGHT_RULES = RuleSet(name="版权合规规则", rules=[
    {"name": "版权状态未知", "status": "未知", "law": "著作权法第24条", "severity": "高", "deduction": 5, "desc": "数据版权状态未知", "action": "确认授权来源"},
    {"name": "疑似侵权", "status": "侵权", "law": "著作权法第47条", "severity": "严重", "deduction": 10, "desc": "数据来源涉嫌侵权", "action": "立即移除并追责"},
])

TRAINING_DATA_RULES = RuleSet(name="训练数据安全规则", rules=[
    {"name": "含PII的训练数据", "law": "生成式AI服务管理暂行办法第7条", "severity": "严重", "deduction": 10, "desc": "含个人信息的未脱敏数据用于训练", "action": "立即隔离，重新脱敏"},
    {"name": "爬虫数据未经授权", "law": "生成式AI服务管理暂行办法第8条", "severity": "高", "deduction": 7, "desc": "未经授权的爬虫数据用于训练", "action": "停止使用，确认授权"},
])


def evaluate_rules(
    dataset: pd.DataFrame,
    decisions: dict = None,
    check_pii: bool = True,
    check_quality: bool = True,
    check_copyright: bool = True,
    check_training: bool = True,
) -> list:
    hits = []
    if decisions is None:
        decisions = {}

    if check_pii and "pii_value" in dataset.columns:
        for _, row in dataset.iterrows():
            if pd.notna(row.get("pii_value")) and isinstance(row["pii_value"], str):
                for rule in PII_RULES.rules:
                    if re.search(rule["pattern"], str(row["pii_value"])):
                        deid_strategy = decisions.get(f"deid_{row.name}", None)
                        if deid_strategy in [None, "不处理"]:
                            hits.append(RuleHit(
                                rule_name=rule["name"], law_article=rule["law"],
                                severity=rule["severity"], deduction=rule["deduction"],
                                description=rule["desc"], suggested_action=rule["action"],
                                record_id=row.get("record_id"), field_name="pii_value",
                            ))

    if check_quality:
        numeric_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            null_rate = dataset[col].isnull().mean()
            if null_rate > DATA_QUALITY_RULES.rules[0]["threshold"]:
                hits.append(RuleHit(
                    rule_name=DATA_QUALITY_RULES.rules[0]["name"],
                    law_article=DATA_QUALITY_RULES.rules[0]["law"],
                    severity=DATA_QUALITY_RULES.rules[0]["severity"],
                    deduction=DATA_QUALITY_RULES.rules[0]["deduction"],
                    description=f"字段'{col}'缺失率 {null_rate:.1%}",
                    suggested_action=DATA_QUALITY_RULES.rules[0]["action"],
                    field_name=col,
                ))
            if dataset[col].std() > 0:
                z_scores = (dataset[col] - dataset[col].mean()) / dataset[col].std()
                outliers = z_scores.abs() > DATA_QUALITY_RULES.rules[2]["threshold"]
                if outliers.any():
                    hits.append(RuleHit(
                        rule_name=DATA_QUALITY_RULES.rules[2]["name"],
                        law_article=DATA_QUALITY_RULES.rules[2]["law"],
                        severity=DATA_QUALITY_RULES.rules[2]["severity"],
                        deduction=DATA_QUALITY_RULES.rules[2]["deduction"] * min(int(outliers.sum()), 5),
                        description=f"字段'{col}'有 {int(outliers.sum())} 个异常值",
                        suggested_action=DATA_QUALITY_RULES.rules[2]["action"],
                        field_name=col,
                    ))

    if check_copyright and "copyright_status" in dataset.columns:
        unknown_mask = dataset["copyright_status"] == COPYRIGHT_RULES.rules[0]["status"]
        infringement_mask = dataset["copyright_status"] == COPYRIGHT_RULES.rules[1]["status"]
        if unknown_mask.any():
            hits.append(RuleHit(
                rule_name=COPYRIGHT_RULES.rules[0]["name"],
                law_article=COPYRIGHT_RULES.rules[0]["law"],
                severity=COPYRIGHT_RULES.rules[0]["severity"],
                deduction=COPYRIGHT_RULES.rules[0]["deduction"],
                description=f"有 {int(unknown_mask.sum())} 条记录版权状态未知",
                suggested_action=COPYRIGHT_RULES.rules[0]["action"],
            ))
        if infringement_mask.any():
            hits.append(RuleHit(
                rule_name=COPYRIGHT_RULES.rules[1]["name"],
                law_article=COPYRIGHT_RULES.rules[1]["law"],
                severity=COPYRIGHT_RULES.rules[1]["severity"],
                deduction=COPYRIGHT_RULES.rules[1]["deduction"] * int(infringement_mask.sum()),
                description=f"有 {int(infringement_mask.sum())} 条记录涉嫌侵权",
                suggested_action=COPYRIGHT_RULES.rules[1]["action"],
            ))

    if check_training:
        training_intended = decisions.get("use_for_training", False)
        if training_intended and "has_pii" in dataset.columns:
            pii_in_training = dataset[dataset["has_pii"] == True]
            if len(pii_in_training) > 0:
                hits.append(RuleHit(
                    rule_name=TRAINING_DATA_RULES.rules[0]["name"],
                    law_article=TRAINING_DATA_RULES.rules[0]["law"],
                    severity=TRAINING_DATA_RULES.rules[0]["severity"],
                    deduction=TRAINING_DATA_RULES.rules[0]["deduction"],
                    description=f"有 {len(pii_in_training)} 条含PII的记录被用于训练",
                    suggested_action=TRAINING_DATA_RULES.rules[0]["action"],
                ))

    return hits


def apply_hard_limits(hits: list) -> tuple:
    severe_hits = [h for h in hits if h.severity == "严重"]
    if severe_hits:
        reasons = "; ".join([f"[{h.law_article}] {h.description}" for h in severe_hits[:3]])
        return False, f"存在 {len(severe_hits)} 项严重违规：{reasons}"
    return True, ""


if __name__ == "__main__":
    from synthetic_data.deyang_generator import generate_dataset, inject_anomalies

    df = generate_dataset(n_records=50, semester=2026, seed=42)
    df_tainted, orders = inject_anomalies(df, anomaly_rate=0.2, seed=42)

    hits = evaluate_rules(df_tainted)
    print(f"命中 {len(hits)} 条规则:")
    for h in hits:
        print(f"  [{h.severity}] {h.rule_name} ({h.law_article}) -{h.deduction}分: {h.description}")

    passed, reason = apply_hard_limits(hits)
    print(f"\n硬拦截: {'通过' if passed else f'拦截 - {reason}'}")
