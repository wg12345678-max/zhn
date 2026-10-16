"""
随机事件引擎：在数据治理全过程中注入不确定性，训练学生应急响应能力
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import random
import numpy as np


@dataclass
class RandomEvent:
    name: str
    description: str
    trigger_prob: float          # 基础触发概率
    options: List[Dict]          # 学生可选应对方案
    default_outcome: str         # 不处理时的默认后果


# 五类随机事件定义
EVENTS = [
    RandomEvent(
        name="数据泄露",
        description="运维人员误操作将含用户日志的数据包上传至公开存储桶，外部可访问。",
        trigger_prob=0.30,
        options=[
            {"label": "立即召回+安全整改+监管报备", "compliance_delta": -5, "cost_delta": 30000, "desc": "快速止损，合规影响小"},
            {"label": "仅删除公开链接", "compliance_delta": -15, "cost_delta": 5000, "desc": "表面处理，可能已被爬取"},
            {"label": "隐瞒不报", "compliance_delta": -30, "cost_delta": 0, "desc": "一旦被发现，后果严重"},
        ],
        default_outcome="合规度-25，触发网信办约谈，成本+50000",
    ),
    RandomEvent(
        name="政策口径突变",
        description="财政部发布新解释，某项数据资源不再满足无形资产确认条件。",
        trigger_prob=0.25,
        options=[
            {"label": "及时调整会计政策并追溯重述", "compliance_delta": 0, "cost_delta": 10000, "desc": "合规，符合会计谨慎性原则"},
            {"label": "维持原处理，等待明确", "compliance_delta": -12, "cost_delta": 0, "desc": "存在政策风险"},
            {"label": "按新口径重估全部资产", "compliance_delta": +2, "cost_delta": 25000, "desc": "彻底但成本高"},
        ],
        default_outcome="原入表方案失效，需冲回，合规度-10",
    ),
    RandomEvent(
        name="上游供应商暴雷",
        description="外部采购数据供应商被查出数据来源违法，其提供数据全部不可用。",
        trigger_prob=0.25,
        options=[
            {"label": "立即停用+寻找替代源+法律追责", "compliance_delta": 0, "cost_delta": 40000, "desc": "合规止损"},
            {"label": "暂时保留观察", "compliance_delta": -10, "cost_delta": 0, "desc": "风险未消除"},
            {"label": "切换至自采数据", "compliance_delta": +3, "cost_delta": 60000, "desc": "彻底但成本高"},
        ],
        default_outcome="外部采购数据全部失效，可用数据量-30%",
    ),
    RandomEvent(
        name="模型投毒",
        description="检测发现训练集中混入对抗样本，模型输出出现异常偏差。",
        trigger_prob=0.25,
        options=[
            {"label": "隔离污染数据+重新清洗+重训", "compliance_delta": 0, "cost_delta": 50000, "desc": "彻底清洗"},
            {"label": "仅删除明显异常样本", "compliance_delta": -8, "cost_delta": 10000, "desc": "可能残留"},
            {"label": "继续训练观察", "compliance_delta": -20, "cost_delta": 0, "desc": "风险极高"},
        ],
        default_outcome="模型效果-25%，需重训，合规度-10",
    ),
    RandomEvent(
        name="监管约谈",
        description="因前期合规瑕疵，网信办要求限期说明情况并整改。",
        trigger_prob=0.20,
        options=[
            {"label": "主动配合+提交整改报告+建章立制", "compliance_delta": +5, "cost_delta": 20000, "desc": "态度端正，合规加分"},
            {"label": "仅提交说明", "compliance_delta": -5, "cost_delta": 5000, "desc": "整改不彻底"},
            {"label": "消极应对", "compliance_delta": -20, "cost_delta": 0, "desc": "可能被从重处罚"},
        ],
        default_outcome="合规度-15，影响后续业务审批",
    ),
]


def roll_events(
    n_rounds: int = 3,
    base_prob: float = 0.35,
    seed: Optional[int] = None,
) -> List[RandomEvent]:
    """
    按轮次滚动触发随机事件（每轮有 base_prob 概率触发一个）

    参数:
        n_rounds: 模拟轮数
        base_prob: 每轮触发概率
        seed: 随机种子

    返回:
        被触发的事件列表
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    triggered = []
    for _ in range(n_rounds):
        if random.random() < base_prob:
            event = random.choice(EVENTS)
            triggered.append(event)
    return triggered


def apply_response(
    event: RandomEvent,
    choice_index: int,
) -> Dict:
    """
    计算学生应对方案的影响

    参数:
        event: 触发的事件
        choice_index: 学生选择的方案索引（-1 表示不处理）

    返回:
        影响结果字典
    """
    if choice_index < 0 or choice_index >= len(event.options):
        return {
            "choice": "未处理",
            "compliance_delta": -15,
            "cost_delta": 0,
            "desc": event.default_outcome,
        }

    opt = event.options[choice_index]
    return {
        "choice": opt["label"],
        "compliance_delta": opt["compliance_delta"],
        "cost_delta": opt["cost_delta"],
        "desc": opt["desc"],
    }


def event_pool() -> List[RandomEvent]:
    """返回全部事件定义（供前端展示）"""
    return EVENTS


if __name__ == "__main__":
    triggered = roll_events(n_rounds=5, base_prob=0.4, seed=42)
    print(f"本轮触发 {len(triggered)} 个事件:")
    for e in triggered:
        print(f"  • {e.name}: {e.description}")

    if triggered:
        result = apply_response(triggered[0], 0)
        print(f"\n应对: {result['choice']}")
        print(f"合规变化: {result['compliance_delta']:+d}, 成本变化: {result['cost_delta']:+d}")
        print(f"说明: {result['desc']}")
