"""
能力画像与报告导出模块
- 四维雷达图（Streamlit 中调用）
- 岗位映射建议
- Markdown 决策备忘录导出
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


def _import_plotly():
    """延迟导入 plotly，避免无 plotly 环境导致模块无法加载。"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        return px, go
    except ImportError:
        return None, None


ABILITY_NAMES = {
    "data_quality": "数据质量能力",
    "compliance": "合规判断能力",
    "asset_operation": "资产运营能力",
    "emergency_response": "应急响应能力",
}


ROLE_MAPPING = [
    {"ability": "compliance", "threshold": 85, "role": "数据合规官"},
    {"ability": "data_quality", "threshold": 85, "role": "数据质量工程师"},
    {"ability": "asset_operation", "threshold": 80, "role": "数据资产估值师"},
    {"ability": "emergency_response", "threshold": 80, "role": "数据安全应急响应专员"},
]


def render_radar(abilities: dict, title: str = "能力画像雷达图") -> Optional["go.Figure"]:  # type: ignore[name-defined]
    """
    生成四维能力雷达图（供 Streamlit 用 st.plotly_chart 展示）

    参数:
        abilities: {"data_quality": 82.1, "compliance": 91.0, ...}
        title: 图表标题

    返回:
        plotly Figure 对象；若环境未安装 plotly 则返回 None
    """
    _, go = _import_plotly()
    if go is None:
        return None

    labels = [ABILITY_NAMES[k] for k in abilities.keys() if k in ABILITY_NAMES]
    values = [abilities[k] for k in abilities.keys() if k in ABILITY_NAMES]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="能力值",
        line=dict(color="#1f77b4", width=2),
        fillcolor="rgba(31, 119, 180, 0.3)",
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        width=500,
        height=500,
    )
    return fig


def map_to_roles(abilities: dict) -> List[str]:
    """根据能力值映射可胜任岗位"""
    matched = []
    for m in ROLE_MAPPING:
        key = m["ability"]
        if key in abilities and abilities[key] >= m["threshold"]:
            matched.append(m["role"])
    return matched if matched else ["数据治理助理（建议继续提升综合能力）"]


def generate_memo(
    group_name: str,
    semester: int,
    abilities: dict,
    grades: dict,
    valuation,
    recognition: Optional[dict] = None,
    training_review: Optional[dict] = None,
    rule_hits: Optional[List] = None,
) -> str:
    """生成 Markdown 格式的决策备忘录"""
    lines = [
        f"# 涉税合规决策备忘录" if False else f"# 数据资产化与合规治理决策备忘录",
        f"**小组名**：{group_name}　　**学期**：{semester}",
        "",
        "## 一、能力画像",
        "",
        "| 能力维度 | 得分 | 等级 |",
        "|---|---|---|",
    ]
    for key, name in ABILITY_NAMES.items():
        score = abilities.get(key, 0)
        grade = grades.get(key, "D")
        lines.append(f"| {name} | {score:.1f} | {grade} |")

    lines += [
        "",
        "## 二、资产估值",
        "",
        f"- 基础价值：{valuation.base_value:,.0f} 元",
        f"- 质量溢价：{valuation.quality_premium:,.0f} 元",
        f"- 风险折损：{valuation.risk_discount:,.0f} 元",
        f"- **最终估值：{valuation.final_value:,.0f} 元**",
        f"- 治理成本：{valuation.governance_cost:,.0f} 元",
        f"- ROI：{valuation.roi_percent}",
        "",
        "## 三、入表判断",
        "",
    ]
    if recognition:
        for c in recognition.get("conditions", []):
            icon = "✓" if c["passed"] else "✗"
            lines.append(f"- [{icon}] **{c['name']}**：{c['detail']}（阈值 {c['threshold']}）")
        lines.append(f"\n**结论**：{'全部满足，可入表' if recognition.get('all_passed') else '存在未满足条件，不可入表'}")

    lines += [
        "",
        "## 四、训练集准入",
        "",
    ]
    if training_review:
        for r in training_review.get("reviews", []):
            icon = "✓" if r["passed"] else "✗"
            lines.append(f"- [{icon}] **{r['name']}**（{r['basis']}）：{r['detail']}")
        lines.append(f"\n**结论**：{'可以进入训练集' if training_review.get('can_train') else '禁止进入训练集'}")

    if rule_hits:
        lines += [
            "",
            "## 五、规则命中记录",
            "",
            "| 规则 | 法条 | 严重级别 | 扣分 | 描述 |",
            "|---|---|---|---|---|",
        ]
        for h in rule_hits[:20]:
            lines.append(f"| {h.rule_name} | {h.law_article} | {h.severity} | -{h.deduction} | {h.description} |")

    lines += ["", "---", "*本备忘录由虚拟仿真实验系统自动生成*"]
    return "\n".join(lines)


if __name__ == "__main__":
    abilities = {"data_quality": 82.1, "compliance": 91.0, "asset_operation": 68.0, "emergency_response": 78.0}
    grades = {"data_quality": "B+", "compliance": "A", "asset_operation": "C+", "emergency_response": "B"}
    print("岗位映射:", map_to_roles(abilities))
    fig = render_radar(abilities)
    print("雷达图对象:", type(fig).__name__)
