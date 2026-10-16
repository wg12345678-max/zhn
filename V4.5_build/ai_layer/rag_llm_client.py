"""
大模型接口 + 降级兜底（LLM 分支）
- 支持腾讯混元 / 智谱 GLM / 本地规则引擎三种模式
- 无 API Key 或网络异常时自动降级为规则引擎，保证系统永远可用
- 知识库由法规条文 + 产业案例 + 教材要点构成（内置，无需外部向量库）
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import json


@dataclass
class LLMResponse:
    text: str
    source: str           # "hunyuan" / "glm" / "rule_fallback"
    citations: List[str]  # 引用的法条/规则
    confidence: float     # 置信度 0-1


@dataclass
class KnowledgeItem:
    title: str
    content: str
    article: str          # 法条出处


# ==================== 内置知识库（法规 + 产业案例 + 教材要点）====================
KNOWLEDGE_BASE: List[KnowledgeItem] = [
    KnowledgeItem(
        title="个人信息保护法第13条",
        content="处理个人信息应当具有明确、合理的目的，并应当与处理目的直接相关，采取对个人权益影响最小的方式。处理个人信息应当限于实现处理目的的最小范围。",
        article="《个人信息保护法》第13条",
    ),
    KnowledgeItem(
        title="个人信息保护法第28条",
        content="敏感个人信息是一旦泄露或者非法使用，容易导致自然人的人格尊严受到侵害或者人身、财产安全受到危害的个人信息，包括生物识别、宗教信仰、医疗健康、金融账户、行踪轨迹等信息。处理敏感个人信息应当取得个人的单独同意。",
        article="《个人信息保护法》第28条",
    ),
    KnowledgeItem(
        title="数据安全法第21条",
        content="国家建立数据分类分级保护制度，根据数据在经济社会发展中的重要程度，以及一旦遭到篡改、破坏、泄露或者非法获取、非法利用，对国家安全、公共利益或者个人、组织合法权益造成的危害程度，对数据实行分类分级保护。",
        article="《数据安全法》第21条",
    ),
    KnowledgeItem(
        title="生成式AI服务管理暂行办法第7条",
        content="生成式人工智能服务提供者应当依法开展预训练、优化训练等训练数据处理活动，使用具有合法来源的数据和基础模型；涉及个人信息的，应当取得个人同意或者符合法律、行政法规规定的其他情形。",
        article="《生成式AI服务管理暂行办法》第7条",
    ),
    KnowledgeItem(
        title="生成式AI服务管理暂行办法第8条",
        content="在生成式人工智能技术研发过程中进行数据标注的，应当制定符合本办法要求的清晰、具体、可操作的标注规则；开展数据标注质量评估，抽样核验标注内容的准确性。",
        article="《生成式AI服务管理暂行办法》第8条",
    ),
    KnowledgeItem(
        title="企业数据资源会计处理暂行规定-入表条件",
        content="企业按照企业会计准则相关规定确认为无形资产或存货等资产的数据资源，应当具备可控制、可计量、可预期带来经济利益三个基本特征。数据权属清晰、成本可可靠计量、存在活跃交易市场或可预期经济利益。",
        article="《企业数据资源相关会计处理暂行规定》",
    ),
    KnowledgeItem(
        title="成德高端能源装备产业集群特征",
        content="成德高端能源装备产业集群以成都研发端（东方电气研究院、国机重装研究院）与德阳制造端（东方汽轮机、东方电机、国机重装）协同为典型特征，形成研发设计—核心零部件制造—总装测试—运维服务全链条。",
        article="成德产业白皮书（教学参考）",
    ),
    KnowledgeItem(
        title="数据治理基本原则-最小必要",
        content="数据治理应遵循最小必要原则，采集的数据应与业务目的直接相关且范围最小；用于训练的数据应在训练完成后及时清理或匿名化。",
        article="教材第3章",
    ),
]


class LLMClient:
    """
    大模型客户端（支持混元 / GLM / 规则兜底三模式自动切换）

    优先级: 混元 API > GLM API > 规则引擎兜底
    任何模式失败都会自动降级，保证仿真系统永远可用。
    """

    def __init__(self, provider: str = "auto", fallback: str = "rule"):
        self.provider = provider
        self.fallback = fallback
        self._resolved = self._resolve_provider()

    def _resolve_provider(self) -> str:
        if self.provider == "auto":
            if os.getenv("HUNYUAN_API_KEY"):
                return "hunyuan"
            if os.getenv("ZHIPU_API_KEY"):
                return "glm"
            return self.fallback
        return self.provider

    def ask(
        self,
        prompt: str,
        context_rules: Optional[List[Any]] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        调用大模型生成回答，失败时自动降级为规则引擎

        参数:
            prompt: 学生问题或场景描述
            context_rules: 当前命中的规则列表（用于规则兜底生成回答）
            temperature: 生成温度

        返回:
            LLMResponse: 统一响应对象
        """
        # 尝试调用混元
        if self._resolved == "hunyuan":
            resp = self._call_hunyuan(prompt, temperature)
            if resp:
                return resp
            self._resolved = "glm"  # 降级

        # 尝试调用 GLM
        if self._resolved == "glm":
            resp = self._call_glm(prompt, temperature)
            if resp:
                return resp
            self._resolved = "rule"  # 降级

        # 规则引擎兜底
        return self._rule_fallback(prompt, context_rules)

    def _call_hunyuan(self, prompt: str, temperature: float) -> Optional[LLMResponse]:
        """调用腾讯混元 API（占位实现，实际部署时补充 SDK）"""
        api_key = os.getenv("HUNYUAN_API_KEY")
        if not api_key:
            return None
        try:
            # 实际调用代码（部署时启用）:
            # from tencentcloud.common import credential
            # from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
            # ...
            return None  # 未配置时返回 None 触发降级
        except Exception:
            return None

    def _call_glm(self, prompt: str, temperature: float) -> Optional[LLMResponse]:
        """调用智谱 GLM API（占位实现，实际部署时补充）"""
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            return None
        try:
            # 实际调用代码（部署时启用）:
            # import zhipuai
            # zhipuai.api_key = api_key
            # r = zhipuai.model_api.invoke(...)
            return None  # 未配置时返回 None 触发降级
        except Exception:
            return None

    def _rule_fallback(self, prompt: str, context_rules: Optional[List[Any]] = None) -> LLMResponse:
        """
        规则引擎兜底：基于内置知识库 + 命中规则生成确定性回答
        保证无 API Key 时系统仍可完整运行
        """
        prompt_lower = prompt.lower()

        # 关键词匹配知识库
        matched = []
        for item in KNOWLEDGE_BASE:
            if any(kw in prompt_lower for kw in [
                item.title.lower(),
                item.article.lower(),
                item.content[:10].lower(),
            ]) or _keyword_hit(prompt_lower, item):
                matched.append(item)

        if not matched and context_rules:
            # 根据命中规则反推相关法条
            for rule in context_rules:
                for item in KNOWLEDGE_BASE:
                    if hasattr(rule, "law_article") and rule.law_article and rule.law_article in item.article:
                        matched.append(item)

        if matched:
            text = "【AI 学伴提示】\n\n" + "\n\n".join([f"📖 {m.article}\n{m.content}" for m in matched[:3]])
            citations = [m.article for m in matched[:3]]
            confidence = 0.85
        else:
            text = (
                "【AI 学伴提示】\n\n"
                "当前为规则引擎模式（未接入大模型 API）。建议您查阅教材相关章节，"
                "或配置 HUNYUAN_API_KEY / ZHIPU_API_KEY 环境变量后重启以启用大模型智能引导。\n\n"
                f"您的问题：{prompt[:100]}"
            )
            citations = []
            confidence = 0.5

        return LLMResponse(text=text, source="rule_fallback", citations=citations, confidence=confidence)

    def cite_articles(self, rule_hits: List[Any]) -> List[str]:
        """把规则命中转成法条引用列表"""
        citations = []
        for hit in rule_hits:
            if hasattr(hit, "law_article") and hit.law_article:
                citations.append(f"[{hit.law_article}] {hit.description}")
        return citations

    def status(self) -> str:
        """返回当前生效的 AI 模式"""
        return self._resolved


def _keyword_hit(prompt: str, item: KnowledgeItem) -> bool:
    keywords = ["个人信息", "敏感信息", "数据分类", "数据分级", "合规", "入表",
                "资产化", "训练数据", "脱敏", "数据治理", "知识产权", "版权",
                "成德", "能源装备", "数据泄露", "数据资源", "无形资产"]
    return any(kw in prompt for kw in keywords)


def retrieve(query: str, k: int = 3) -> List[KnowledgeItem]:
    """
    知识库检索（用于实验前推荐 / 实验中引导 / 实验后复盘）
    使用关键词重叠度排序，无需外部向量库依赖
    """
    query_lower = query.lower()
    scored = []
    for item in KNOWLEDGE_BASE:
        text = (item.title + item.content + item.article).lower()
        score = sum(1 for ch in set(query_lower) if ch in text)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


if __name__ == "__main__":
    client = LLMClient(provider="rule")  # 强制规则模式
    print(f"当前模式: {client.status()}")

    resp = client.ask("数据泄露了怎么办", context_rules=[])
    print(f"来源: {resp.source}")
    print(f"置信度: {resp.confidence}")
    print(resp.text)

    print("\n--- 知识库检索测试 ---")
    for item in retrieve("数据资产入表需要什么条件", k=3):
        print(f"  • {item.article}")
