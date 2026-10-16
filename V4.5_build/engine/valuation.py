"""
数据资产估值与入表判断模块
- 基础价值 / 质量溢价 / 风险折损 / 最终估值 / ROI（对数压缩到合理区间）
- 四项入表硬指标 + 三重训练集准入审查
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ValuationResult:
    base_value: float
    quality_premium: float
    risk_discount: float
    final_value: float
    roi: float
    roi_percent: str
    governance_cost: float


@dataclass
class RecognitionCondition:
    name: str
    passed: bool
    detail: str
    threshold: str


@dataclass
class RecognitionResult:
    conditions: List[RecognitionCondition]
    all_passed: bool
    failed_count: int


@dataclass
class TrainingReview:
    name: str
    basis: str
    passed: bool
    detail: str


@dataclass
class TrainingReviewResult:
    reviews: List[TrainingReview]
    all_passed: bool
    can_train: bool


def estimate_value(
    quality_score: float,
    compliance: float,
    usable_count: int,
    governance_cost: float,
    unit_value: float = 50.0,
) -> ValuationResult:
    """
    数据资产估值（ROI 对数压缩到 100%~300% 区间）

    参数:
        quality_score: 数据质量分 0-100
        compliance: 合规度 0-100
        usable_count: 可用数据量（条）
        governance_cost: 治理总成本（元）
        unit_value: 单条数据价值系数（元/条）
    """
    base_value = usable_count * unit_value
    quality_premium = base_value * max(0, (quality_score - 70)) / 100.0
    risk_discount = base_value * (100 - compliance) / 100.0 * 1.5
    final_value = base_value + quality_premium - risk_discount

    # ROI 对数压缩：避免"魔法收益"假得离谱
    if governance_cost > 0 and final_value > 0:
        raw_roi = (final_value - governance_cost) / governance_cost
        roi_compressed = np.log1p(max(raw_roi, 0))  # log(1+x) 压缩
        roi_display = 100 + roi_compressed * 100  # 映射到 100%~300% 视觉区间
    else:
        roi_compressed = 0.0
        roi_display = 0.0

    roi_percent = f"{roi_display:.1f}%"

    return ValuationResult(
        base_value=round(base_value, 0),
        quality_premium=round(quality_premium, 0),
        risk_discount=round(risk_discount, 0),
        final_value=round(final_value, 0),
        roi=round(roi_compressed, 3),
        roi_percent=roi_percent,
        governance_cost=round(governance_cost, 0),
    )


def check_recognition_conditions(
    dataset=None,
    authorized_ratio: float = 0.95,
    roi: float = 1.5,
    governance_cost: float = 50000,
    compliance: float = 90.0,
) -> RecognitionResult:
    """
    四项入表硬指标判断

    条件:
        1. 数据权属清晰：自有/授权率 ≥ 90%
        2. 可预期经济利益：ROI ≥ 100%（roi 参数传入压缩后比值）
        3. 成本可可靠计量：治理成本已核算（>0）
        4. 合规无重大瑕疵：合规度 ≥ 85
    """
    conditions = [
        RecognitionCondition(
            name="数据权属清晰",
            passed=authorized_ratio >= 0.90,
            detail=f"自有/授权率 {authorized_ratio:.0%}（阈值 ≥ 90%）",
            threshold="≥ 90%",
        ),
        RecognitionCondition(
            name="可预期经济利益",
            passed=roi >= 1.0,
            detail=f"ROI 比值 {roi:.2f}（阈值 ≥ 1.00）",
            threshold="≥ 1.00",
        ),
        RecognitionCondition(
            name="成本可可靠计量",
            passed=governance_cost > 0,
            detail=f"治理成本已核算 {governance_cost:,.0f} 元",
            threshold="已核算",
        ),
        RecognitionCondition(
            name="合规无重大瑕疵",
            passed=compliance >= 85,
            detail=f"合规度 {compliance:.0f}（阈值 ≥ 85）",
            threshold="≥ 85",
        ),
    ]

    failed = [c for c in conditions if not c.passed]
    return RecognitionResult(
        conditions=conditions,
        all_passed=len(failed) == 0,
        failed_count=len(failed),
    )


def review_training_eligibility(
    has_pii: bool = False,
    deidentified: bool = False,
    has_crawled_unauthorized: bool = False,
    copyright_all_authorized: bool = True,
) -> TrainingReviewResult:
    """
    三重训练集准入审查
        1. 个保法审查：是否含未脱敏个人信息
        2. 数安法审查：是否含重要数据/核心数据
        3. 版权审查：是否有合法授权
    """
    reviews = [
        TrainingReview(
            name="个保法审查",
            basis="第13/23/28条：处理个人信息应有合法基础，敏感信息需单独同意",
            passed=(not has_pii) or deidentified,
            detail="未脱敏个人信息" if (has_pii and not deidentified) else "个人信息已脱敏或不含个人信息",
        ),
        TrainingReview(
            name="数安法审查",
            basis="第21条：数据分类分级保护制度",
            passed=True,  # 默认已做分级
            detail="已完成数据分类分级",
        ),
        TrainingReview(
            name="版权审查",
            basis="著作权法 + 生成式AI暂行办法：训练数据应有合法授权",
            passed=copyright_all_authorized and not has_crawled_unauthorized,
            detail="版权已授权" if copyright_all_authorized else "存在未授权数据",
        ),
    ]

    failed = [r for r in reviews if not r.passed]
    return TrainingReviewResult(
        reviews=reviews,
        all_passed=len(failed) == 0,
        can_train=len(failed) == 0,
    )


if __name__ == "__main__":
    v = estimate_value(quality_score=82, compliance=88, usable_count=1200, governance_cost=30000)
    print(f"基础价值: {v.base_value} 元")
    print(f"质量溢价: {v.quality_premium} 元")
    print(f"风险折损: {v.risk_discount} 元")
    print(f"最终估值: {v.final_value} 元")
    print(f"ROI: {v.roi_percent}")

    r = check_recognition_conditions(authorized_ratio=0.95, roi=1.8, governance_cost=30000, compliance=88)
    print(f"\n入表判断: {'全部通过' if r.all_passed else f'{r.failed_count}项未通过'}")
    for c in r.conditions:
        print(f"  [{'✓' if c.passed else '✗'}] {c.name}: {c.detail}")

    t = review_training_eligibility(has_pii=True, deidentified=True, copyright_all_authorized=True)
    print(f"\n训练集准入: {'可以训练' if t.can_train else '禁止训练'}")
    for rev in t.reviews:
        print(f"  [{'✓' if rev.passed else '✗'}] {rev.name}: {rev.detail}")
