"""
面向大模型应用企业的数据资产化与合规治理虚拟仿真实验 —— 主程序入口

教学场景：成德高端能源装备企业"研发在成都、制造在德阳"跨域数据流转

运行: streamlit run main_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ==================== 模块导入 ====================
# 说明：所有对象均为 @dataclass，一律使用 . 属性访问，禁止使用 [] 下标访问
from synthetic_data.deyang_generator import generate_dataset, inject_anomalies, DEFAULT_SOURCE_WEIGHTS
from engine.rule_engine import evaluate_rules, apply_hard_limits, RuleHit
from engine.valuation import estimate_value, check_recognition_conditions, review_training_eligibility
from engine.event_engine import roll_events, apply_response
from audit.audit_logger import AuditLogger, get_logger
from audit.scorer import score_session, compute_abilities, Abilities
from reporting.ability_portrait import render_radar, map_to_roles, generate_memo
from ai_layer.rag_llm_client import LLMClient, retrieve
from database import init_db, create_students, generate_200_accounts, verify_login, save_session, finish_session

st.set_page_config(page_title="数据资产化与合规治理虚拟仿真", page_icon="🛡️", layout="wide")

# ==================== 初始化 ====================
init_db()
DATA_DIR = "./data_cache"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("./audit_logs", exist_ok=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{datetime.now().strftime('%Y%m%d%H%M%S')}"
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "logger" not in st.session_state:
    st.session_state.logger = get_logger(st.session_state.session_id)
if "completed_steps" not in st.session_state:
    st.session_state.completed_steps = set()

logger: AuditLogger = st.session_state.logger

# ==================== 侧边栏 ====================
st.sidebar.title("🛡️ 数据治理仿真")
st.sidebar.caption("成德高端能源装备 · 跨域数据流转")

with st.sidebar.expander("登录（可选，200个测试账号）", expanded=False):
    st.caption("用户名 dg001~dg200，密码见 accounts.csv")
    u = st.text_input("用户名")
    p = st.text_input("密码", type="password")
    if st.button("登录"):
        name = verify_login(u, p)
        if name:
            st.success(f"已登录：{name}")
            st.session_state.username = u
        else:
            st.error("账号或密码错误")

group_name = st.sidebar.text_input("小组名", value="第1组")
semester = st.sidebar.selectbox("学期", [2024, 2025, 2026], index=2)
clean_intensity = st.sidebar.slider("清洗强度", 0.0, 1.0, 0.6, 0.05)
compliance_strict = st.sidebar.slider("合规严格度", 0.0, 1.0, 0.5, 0.05)
biz_aggressiveness = st.sidebar.slider("业务激进度", 0.0, 1.0, 0.5, 0.05)
asset_op_intensity = st.sidebar.slider("资产运营强度", 0.0, 1.0, 0.5, 0.05)

ai_mode = st.sidebar.selectbox(
    "AI学伴模式", ["rule_fallback", "auto"], index=0,
    help="rule_fallback=规则兜底(无API), auto=自动检测API Key"
)
llm_client = LLMClient(provider=ai_mode)
st.sidebar.caption(f"当前AI模式: {llm_client.status()}")

# ==================== 主界面 ====================
st.title("面向大模型应用企业的数据资产化与合规治理虚拟仿真实验")
st.markdown(f"**小组**：{group_name}　**学期**：{semester}　**AI模式**：{llm_client.status()}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📖 实验导览", "🔍 模块一·数据体检", "⚖️ 模块二·合规博弈", "💰 模块三·资产判断", "📊 能力画像"]
)

# ---- 实验导览 ----
with tab1:
    st.header("实验导览：成德高端能源装备企业跨域数据流转")
    st.markdown("""
    ### 你的角色
    你将扮演一家成德高端能源装备企业的 **数据治理负责人（CDO）**，
    管理"成都研发端 — 德阳制造端 — 全国售后端"的跨域数据流转。

    ### 三阶任务
    1. **模块一·数据体检**：对多源数据进行清洗、脱敏、分类分级，生成异常工单
    2. **模块二·合规博弈**：在合规、业务、财务、算法四方利益冲突中做权衡决策，应对突发事件
    3. **模块三·资产判断**：完成数据资产估值、入表条件判断、训练集准入审查

    ### 评分机制
    全程 20 步交互逐项赋分，形成四维能力画像：**数据质量 / 合规判断 / 资产运营 / 应急响应**

    ### 设计理念
    本实验以成德高端能源装备产业集群为产业依托，覆盖《个人信息保护法》《数据安全法》
    《生成式AI服务管理暂行办法》《企业数据资源会计处理暂行规定》等法规要求。
    """)
    if st.button("确认开始实验（记录步骤1）"):
        logger.log_step(1, "确认开始实验", {"group": group_name, "semester": semester})
        st.session_state.completed_steps.add(1)
        st.success("实验已启动，审计日志开始记录")

# ---- 模块一 ----
with tab2:
    st.header("模块一：多源数据体检与合规预处理")
    n_records = st.slider("生成数据量", 100, 2000, 500, 100)
    anomaly_rate = st.slider("异常比例", 0.05, 0.4, 0.15, 0.05)

    if st.button("① 生成成德合成数据集", key="gen"):
        df = generate_dataset(n_records=n_records, semester=semester, seed=42)
        st.session_state.dataset = df
        logger.log_step(5, "生成合成数据集", {"n_records": n_records, "semester": semester}, {"rows": len(df)})
        st.success(f"生成 {len(df)} 条记录，覆盖 5 类数据源")
        st.bar_chart(df["data_source"].value_counts())

    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        if st.button("② 注入异常并生成工单", key="anomaly"):
            df_t, orders = inject_anomalies(df, anomaly_rate=anomaly_rate, seed=42)
            st.session_state.dataset_tainted = df_t
            st.session_state.work_orders = orders
            logger.log_step(6, "生成异常工单", {"anomaly_rate": anomaly_rate}, {"n_orders": len(orders)})
            st.success(f"生成 {len(orders)} 个异常工单")

        if "work_orders" in st.session_state:
            orders = st.session_state.work_orders
            st.subheader("异常工单明细")
            wdf = pd.DataFrame(orders)
            st.dataframe(wdf, use_container_width=True)

            st.subheader("工单处理决策")
            cols = st.columns(3)
            with cols[0]:
                deid = st.selectbox("敏感字段脱敏策略", ["掩码", "泛化", "加密", "差分隐私", "不处理"])
            with cols[1]:
                class_lvl = st.selectbox("数据分类分级", ["公开", "内部", "机密", "绝密"])
            with cols[2]:
                training = st.checkbox("用于大模型训练", value=False)

            decisions = {
                "deid_strategy": deid,
                "classification": class_lvl,
                "use_for_training": training,
            }

            if st.button("③ 运行规则检测", key="rule"):
                df_t = st.session_state.dataset_tainted
                # 把脱敏决策注入数据集：若选择了脱敏策略，则标记已脱敏
                if deid != "不处理" and "has_pii" in df_t.columns:
                    df_t = df_t.copy()
                    df_t.loc[df_t["has_pii"] == True, "pii_value"] = "[已脱敏]"

                hits = evaluate_rules(df_t, decisions=decisions)
                st.session_state.rule_hits = hits
                for h in hits:
                    # RuleHit 为 @dataclass，使用 . 属性访问
                    logger.log_rule_hit({
                        "rule_name": h.rule_name,
                        "law_article": h.law_article,
                        "severity": h.severity,
                        "deduction": h.deduction,
                        "description": h.description,
                        "suggested_action": h.suggested_action,
                    })
                logger.log_step(8, "选择脱敏策略", {"strategy": deid}, {"hits": len(hits)})

                passed, reason = apply_hard_limits(hits)
                st.session_state.hard_passed = passed
                st.session_state.hard_reason = reason

                if passed:
                    st.success(f"规则检测通过，命中 {len(hits)} 条需关注规则")
                else:
                    st.error(f"红线规则硬拦截：{reason}")

                if hits:
                    hdf = pd.DataFrame([{
                        "规则": h.rule_name, "法条": h.law_article,
                        "级别": h.severity, "扣分": h.deduction, "描述": h.description
                    } for h in hits])
                    st.dataframe(hdf, use_container_width=True)

            if st.button("④ 咨询AI学伴", key="ask1"):
                hits = st.session_state.get("rule_hits", [])
                resp = llm_client.ask("数据治理中常见的合规风险有哪些", context_rules=hits)
                st.info(f"来源: {resp.source} | 置信度: {resp.confidence}")
                st.write(resp.text)

# ---- 模块二 ----
with tab3:
    st.header("模块二：四方合规博弈与风险演化")
    st.caption("合规官 / 业务负责人 / 财务总监 / 算法总监 四方KPI冲突")

    if st.button("⑤ 运行合规博弈计算", key="game"):
        # 核心联动公式
        compliance = max(0.0, min(100.0, 100 - (1 - compliance_strict) * 60 - clean_intensity * 10))
        biz_avail = max(10.0, 100 - compliance_strict * 40 - clean_intensity * 20)
        asset_val = max(0.0, biz_avail * 100 * asset_op_intensity - clean_intensity * 50000)
        algo_eff = max(0.0, 100 - (1 - clean_intensity) * 30 - (1 - compliance_strict) * 15)

        st.session_state.game_result = {
            "compliance": compliance, "biz_avail": biz_avail,
            "asset_val": asset_val, "algo_eff": algo_eff,
        }
        logger.log_step(11, "调整三方参数", {
            "compliance_strict": compliance_strict,
            "biz_aggressiveness": biz_aggressiveness,
            "asset_op_intensity": asset_op_intensity,
        }, {"compliance": compliance, "biz_avail": biz_avail})

        st.metric("合规度", f"{compliance:.1f}")
        st.metric("业务可用性", f"{biz_avail:.1f}")
        st.metric("资产运营价值", f"{asset_val:,.0f} 元")
        st.metric("算法效果", f"{algo_eff:.1f}")

        kdf = pd.DataFrame({
            "角色": ["合规官", "业务负责人", "财务总监", "算法总监"],
            "KPI": [compliance, biz_avail, min(100.0, asset_val / 1000), algo_eff],
        })
        fig = px.bar(kdf, x="角色", y="KPI", title="四方KPI对比")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("随机事件应对")
    if st.button("⑥ 触发随机事件", key="event"):
        events = roll_events(n_rounds=3, base_prob=0.4, seed=semester)
        st.session_state.events = events
        logger.log_step(13, "触发随机事件", {}, {"n_events": len(events)})
        for e in events:
            st.warning(f"**{e.name}**：{e.description}")
            options_labels = [o["label"] for o in e.options]
            choice = st.radio(f"应对方案 - {e.name}", options_labels + ["不处理"], key=e.name)
            idx = options_labels.index(choice) if choice in options_labels else -1
            result = apply_response(e, idx)
            logger.log_random_event(e.name, result["choice"], result["desc"])
            st.info(f"结果：{result['desc']}（合规{result['compliance_delta']:+d}，成本{result['cost_delta']:+d}）")

# ---- 模块三 ----
with tab4:
    st.header("模块三：数据资产估值与入表判断")
    if st.button("⑦ 运行资产估值", key="val"):
        gr = st.session_state.get("game_result")
        quality = gr["biz_avail"] if gr else 75.0
        compliance = gr["compliance"] if gr else 80.0
        usable = int(n_records * 0.7)
        cost = clean_intensity * 100000 + 30000

        v = estimate_value(quality, compliance, usable, cost)
        st.session_state.valuation = v
        logger.log_step(15, "运行资产估值", {"quality": quality, "compliance": compliance, "usable": usable},
                       {"final_value": v.final_value, "roi": v.roi_percent})
        st.metric("基础价值", f"{v.base_value:,.0f} 元")
        st.metric("质量溢价", f"{v.quality_premium:,.0f} 元")
        st.metric("风险折损", f"{v.risk_discount:,.0f} 元")
        st.metric("最终估值", f"{v.final_value:,.0f} 元", delta=v.roi_percent)

    if "valuation" in st.session_state:
        v = st.session_state.valuation
        st.subheader("入表条件判断")
        r = check_recognition_conditions(
            authorized_ratio=0.95, roi=v.roi, governance_cost=v.governance_cost, compliance=compliance_strict * 100
        )
        # 重要：RecognitionCondition 是 @dataclass，一律使用 . 属性访问，禁止 c['threshold']
        for c in r.conditions:
            icon = "✅" if c.passed else "❌"
            st.write(f"{icon} **{c.name}**：{c.detail}（阈值 {c.threshold}）")
        # 转存到 session_state 时也使用 __dict__（已经是标准做法）
        st.session_state.recognition = {
            "conditions": [c.__dict__ for c in r.conditions],
            "all_passed": r.all_passed,
        }

        st.subheader("训练集准入审查")
        has_pii = st.session_state.get("dataset_tainted") is not None
        # deid 变量来自模块一的 selectbox，若未定义则默认为"不处理"
        try:
            deid_val = deid
        except NameError:
            deid_val = "不处理"
        t = review_training_eligibility(
            has_pii=has_pii,
            deidentified=(deid_val != "不处理"),
            copyright_all_authorized=True,
        )
        # TrainingReview 同为 @dataclass，使用 . 属性访问
        for rev in t.reviews:
            icon = "✅" if rev.passed else "❌"
            st.write(f"{icon} **{rev.name}**（{rev.basis}）：{rev.detail}")
        st.session_state.training_review = {
            "reviews": [rev.__dict__ for rev in t.reviews],
            "can_train": t.can_train,
        }

    if st.button("⑧ 生成决策备忘录", key="memo"):
        abilities_dummy = {"data_quality": 82.1, "compliance": 91.0, "asset_operation": 68.0, "emergency_response": 78.0}
        grades_dummy = {"data_quality": "B+", "compliance": "A", "asset_operation": "C+", "emergency_response": "B"}
        v = st.session_state.get("valuation")
        memo = generate_memo(
            group_name, semester, abilities_dummy, grades_dummy, v,
            st.session_state.get("recognition"), st.session_state.get("training_review"),
            st.session_state.get("rule_hits", [])
        )
        st.download_button("下载备忘录 (Markdown)", memo, file_name=f"{group_name}_备忘录.md", mime="text/markdown")
        logger.log_step(20, "导出实验报告", {"group": group_name}, {})
        st.success("备忘录已生成")

# ---- 能力画像 ----
with tab5:
    st.header("能力画像与审计日志")
    if st.button("⑨ 生成本次能力画像", key="portrait"):
        audit_df = logger.get_audit_trail()
        if audit_df.empty:
            st.warning("暂无操作记录，请先完成实验步骤")
        else:
            score = score_session(audit_df, st.session_state.session_id)
            abilities = compute_abilities(score, audit_df)
            st.session_state.abilities = abilities.to_dict()
            st.session_state.grades = abilities.grades()

            col1, col2 = st.columns(2)
            with col1:
                fig = render_radar(abilities.to_dict())
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric("总分", f"{score.total_final:.1f} / {score.total_max}")
                st.metric("完成度", f"{score.percentage:.1f}%")
                st.write("**等级**", abilities.grades())
                st.write("**可胜任岗位**", map_to_roles(abilities.to_dict()))

            st.subheader("逐条赋分明细")
            sdf = pd.DataFrame([{
                "步骤": s.step_no, "操作": s.action, "满分": s.max_score,
                "得分": s.final_score, "扣分明细": "; ".join([f"{k}(-{v})" for k, v in s.deductions])
            } for s in score.step_scores])
            st.dataframe(sdf, use_container_width=True)

    if st.button("⑩ 导出审计日志 CSV", key="export"):
        path = logger.export_csv()
        with open(path, "r", encoding="utf-8") as f:
            st.download_button("下载审计日志", f.read(), file_name=os.path.basename(path), mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("V4.2 · 规则引擎版（可无缝切换 LLM）")
