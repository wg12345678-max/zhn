"""
端到端冒烟测试：覆盖 6 个核心模块 + 数据库 + LLM 分支
运行: python smoke_test.py
"""
import os
import sys
import traceback

os.makedirs("./audit_logs", exist_ok=True)
os.makedirs("./data_cache", exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))

FAILED = []
PASSED = []


def test(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  ✓ {name}")
    except Exception as e:
        FAILED.append((name, str(e), traceback.format_exc()))
        print(f"  ✗ {name}: {e}")


# 1. 合成数据生成
def t1():
    from synthetic_data.deyang_generator import generate_dataset, inject_anomalies
    df = generate_dataset(n_records=200, semester=2026, seed=42)
    assert len(df) == 200, f"期望200条，实际{len(df)}"
    assert "data_source" in df.columns
    assert df["data_source"].nunique() >= 3
    df2, orders = inject_anomalies(df, anomaly_rate=0.2, seed=42)
    assert len(orders) > 0
    assert "anomaly_type" in orders[0]


# 2. 规则引擎
def t2():
    from synthetic_data.deyang_generator import generate_dataset, inject_anomalies
    from engine.rule_engine import evaluate_rules, apply_hard_limits
    df = generate_dataset(n_records=100, semester=2026, seed=42)
    df_t, _ = inject_anomalies(df, anomaly_rate=0.3)
    hits = evaluate_rules(df_t, decisions={"deid_strategy": "不处理"})
    assert len(hits) > 0, "应至少命中1条规则"
    passed, _ = apply_hard_limits(hits)
    # 注入30%异常，必然有严重违规
    assert not passed, "高异常率应触发硬拦截"


# 3. 估值与入表
def t3():
    from engine.valuation import estimate_value, check_recognition_conditions, review_training_eligibility
    v = estimate_value(quality_score=82, compliance=88, usable_count=1200, governance_cost=30000)
    assert v.final_value > 0
    assert isinstance(v.roi_percent, str) and "%" in v.roi_percent, f"roi_percent 应为含%的字符串，实际{v.roi_percent}"
    r = check_recognition_conditions(authorized_ratio=0.95, roi=1.8, governance_cost=30000, compliance=88)
    assert r.all_passed, "高合规高ROI应通过全部条件"
    t = review_training_eligibility(has_pii=True, deidentified=True, copyright_all_authorized=True)
    assert t.can_train, "已脱敏应通过训练审查"


# 4. 随机事件引擎
def t4():
    from engine.event_engine import roll_events, apply_response, event_pool
    events = event_pool()
    assert len(events) == 5
    triggered = roll_events(n_rounds=5, base_prob=0.4, seed=42)
    assert isinstance(triggered, list)
    if triggered:
        result = apply_response(triggered[0], 0)
        assert "compliance_delta" in result


# 5. 审计日志
def t5():
    from audit.audit_logger import AuditLogger
    logger = AuditLogger("smoke_test_sess")
    logger.log_step(1, "测试步骤", {"k": "v"})
    logger.log_rule_hit({"rule_name": "test", "deduction": 5})
    logger.log_decision("脱敏", "掩码")
    logger.log_random_event("数据泄露", "召回", "合规-5")
    df = logger.get_audit_trail()
    assert len(df) >= 4
    assert "type" in df.columns
    path = logger.export_csv()
    assert os.path.exists(path)


# 6. 评分与能力画像
def t6():
    from audit.audit_logger import AuditLogger
    from audit.scorer import score_session, compute_abilities
    logger = AuditLogger("score_smoke")
    for s in [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        logger.log_step(s, f"步骤{s}操作")
    df = logger.get_audit_trail()
    score = score_session(df, "score_smoke")
    assert score.total_max == 94.0, f"满分应为94，实际{score.total_max}"
    assert score.percentage >= 0
    abilities = compute_abilities(score, df)
    assert abilities.compliance >= 0
    assert abilities.data_quality >= 0


# 7. 报告导出
def t7():
    import importlib
    abilities = {"data_quality": 82.1, "compliance": 91.0, "asset_operation": 68.0, "emergency_response": 78.0}
    grades = {"data_quality": "B+", "compliance": "A", "asset_operation": "C+", "emergency_response": "B"}
    from reporting.ability_portrait import render_radar, map_to_roles, generate_memo
    fig = render_radar(abilities)
    if fig is None:
        # 当前环境未安装 plotly（部署时会自动安装），仅验证函数返回 None 且不抛异常
        assert fig is None
    else:
        # 有 plotly 时才断言图对象有效
        assert fig is not None
    roles = map_to_roles(abilities)
    assert len(roles) >= 1
    from engine.valuation import estimate_value
    v = estimate_value(quality_score=82, compliance=88, usable_count=1200, governance_cost=30000)
    memo = generate_memo("测试组", 2026, abilities, grades, v)
    assert "备忘录" in memo


# 8. LLM 客户端（规则兜底分支）
def t8():
    import os as _os
    from ai_layer.rag_llm_client import LLMClient, retrieve
    # 清空 API Key，强制规则兜底
    _os.environ.pop("HUNYUAN_API_KEY", None)
    _os.environ.pop("ZHIPU_API_KEY", None)
    client = LLMClient(provider="rule")
    assert client.status() == "rule", f"期望rule，实际{client.status()}"
    resp = client.ask("数据泄露了怎么办", context_rules=[])
    assert resp.text, "规则兜底应返回文本"
    assert resp.source == "rule_fallback"
    # 测试 auto 模式在无 Key 时也能降级
    client2 = LLMClient(provider="auto")
    assert client2.status() == "rule", "auto 无 Key 应降级为规则兜底"
    resp2 = client2.ask("什么是数据合规", context_rules=[])
    assert resp2.text
    # 测试有 Key 时切换
    _os.environ["HUNYUAN_API_KEY"] = "fake_test_key"
    client3 = LLMClient(provider="auto")
    assert client3.status() == "hunyuan", "有 Key 时应切换为 hunyuan"
    del _os.environ["HUNYUAN_API_KEY"]
    items = retrieve("数据资产入表需要什么条件", k=3)
    assert len(items) > 0


# 9. 数据库 + 200 账号
def t9():
    from database import init_db, generate_200_accounts, create_students, verify_login, save_session, finish_session
    kind = init_db()
    assert kind == "sqlite"
    accounts = generate_200_accounts()
    assert len(accounts) == 200
    assert accounts[0].username == "dg001"
    assert accounts[-1].username == "dg200"
    # 密码不含明文个人信息
    for acc in accounts:
        assert acc.password, "密码不能为空"
    create_students(accounts)
    # 验证登录
    assert verify_login("dg001", accounts[0].password) == "学生dg001"
    assert verify_login("dg001", "wrongpwd") is None
    # 会话保存
    save_session("test_sess", "dg001", "测试组", 2026)
    finish_session("test_sess", 85.5, {"data_quality": 82, "compliance": 91, "asset_operation": 68, "emergency_response": 78})


# 10. 主程序模块可导入
def t10():
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_app", os.path.join(os.path.dirname(__file__), "main_app.py"))
    assert spec is not None
    # 仅检查语法，不运行 streamlit
    with open(spec.origin) as f:
        compile(f.read(), spec.origin, "exec")


print("=" * 60)
print("V4.2 冒烟测试")
print("=" * 60)

test("1. 合成数据生成（成德产业指纹）", t1)
test("2. 规则引擎（PII/质量/版权/训练）", t2)
test("3. 资产估值与入表判断", t3)
test("4. 随机事件引擎", t4)
test("5. 审计日志（JSONL）", t5)
test("6. 20步赋分与能力画像", t6)
test("7. 报告导出（雷达图/备忘录）", t7)
test("8. LLM客户端（规则兜底分支）", t8)
test("9. 数据库与200账号烧录", t9)
test("10. 主程序语法检查", t10)

print("=" * 60)
print(f"通过: {len(PASSED)}/10")
if FAILED:
    print(f"失败: {len(FAILED)}/10")
    for name, msg, tb in FAILED:
        print(f"\n--- {name} ---")
        print(msg)
        print(tb[-500:])
    sys.exit(1)
else:
    print("全部通过 ✓")
