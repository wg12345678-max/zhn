"""
成德高端能源装备企业跨域数据流转 - 合成数据生成器 V4.2
功能：生成带产业指纹的合成数据，含5类数据源、PII脱敏标记、异常注入
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ==================== 配置参数 ====================
DATA_SOURCES = [
    "研发设计图纸",
    "生产工艺参数",
    "供应链物料清单",
    "售后服务记录",
    "能耗监测数据"
]

PII_TYPES = ["身份证", "手机号", "邮箱", "地址", "银行卡号"]

# 默认数据源权重（供旧版调用 / 向后兼容）
DEFAULT_SOURCE_WEIGHTS = {
    "研发设计图纸": 0.25,
    "生产工艺参数": 0.20,
    "供应链物料清单": 0.20,
    "售后服务记录": 0.20,
    "能耗监测数据": 0.15,
}

# 成德高端能源装备产业关键词（产业指纹）
DEYANG_INDUSTRY_TAGS = {
    "研发设计图纸": ["汽轮机叶片", "燃气轮机", "风电齿轮箱", "水轮机转轮", "核电主泵"],
    "生产工艺参数": ["焊接工艺WPS", "热处理曲线", "铸造砂型", "锻压比", "涂层厚度"],
    "供应链物料清单": ["合金钢锭", "精密轴承", "密封件", "液压阀组", "传感器模组"],
    "售后服务记录": ["故障代码F01-F99", "维修工单", "备件更换记录", "巡检报告", "远程诊断日志"],
    "能耗监测数据": ["电耗(kWh)", "天然气(m³)", "压缩空气(m³/h)", "水循环量(t)", "蒸汽压力(MPa)"]
}

# 四川地区常用身份证前缀（地市码）
_AREA_CODES = [
    "510101", "510106", "510107", "510112", "510113", "510114", "510115",
    "510116", "510117", "510118", "510121", "510129", "510131", "510181",
    "510182", "510183", "510184", "510302", "510303", "510304", "510311",
    "510321", "510322", "510323", "510324", "510421", "510422", "510423",
    "510424", "510425", "510501", "510502", "510503", "510504", "510521",
    "510522", "510523", "510524", "510525", "510601", "510602", "510603",
    "510604", "510621", "510622", "510623", "510681", "510682", "510683",
    "510701", "510702", "510703", "510704", "510705", "510721", "510722",
    "510723", "510724", "510725", "510726", "510727", "510781", "510801",
    "510802", "510811", "510812", "510821", "510822", "510823", "510824",
    "510901", "510902", "510903", "510904", "510921", "510922", "510923",
]

# ==================== PII 生成函数（纯字符串构建，无 int32 溢出风险） ====================

def _gen_pii(pii_type: str) -> str:
    """
    生成指定类型的PII数据（纯字符串构建，避免int32溢出）

    参数:
        pii_type: PII类型（身份证/手机号/邮箱/地址/银行卡号）

    返回:
        模拟的PII字符串
    """
    if pii_type == "身份证":
        area = np.random.choice(_AREA_CODES)
        year = f"{np.random.randint(1960, 2005):04d}"
        month = f"{np.random.randint(1, 13):02d}"
        day = f"{np.random.randint(1, 28):02d}"
        birth = year + month + day
        seq = f"{np.random.randint(0, 10000):04d}"
        check = str(np.random.randint(0, 10))
        return area + birth + seq + check

    elif pii_type == "手机号":
        prefixes = [
            "138", "139", "150", "151", "152", "157", "158", "159",
            "182", "183", "187", "188", "130", "131", "132", "155",
            "156", "186", "185", "189", "133", "153", "177", "180",
            "181", "134", "135", "136", "137"
        ]
        prefix = np.random.choice(prefixes)
        suffix = f"{np.random.randint(0, 100000000):08d}"
        return prefix + suffix

    elif pii_type == "邮箱":
        letters = list("abcdefghijklmnopqrstuvwxyz")
        name_len = np.random.randint(6, 12)
        name = ''.join(np.random.choice(letters, size=name_len))
        domains = ["qq.com", "163.com", "gmail.com", "outlook.com", "company.cn", "dyee.com"]
        domain = np.random.choice(domains)
        return f"{name}@{domain}"

    elif pii_type == "地址":
        cities = ["成都市", "德阳市", "绵阳市", "眉山市"]
        districts = ["高新区", "锦江区", "旌阳区", "涪城区", "东坡区"]
        streets = ["天府大道", "剑南大道", "泰山路", "岷江路", "长江路"]
        numbers = [f"{i}号" for i in range(1, 51)]
        return (np.random.choice(cities) +
                np.random.choice(districts) +
                np.random.choice(streets) +
                np.random.choice(numbers))

    elif pii_type == "银行卡号":
        bin_prefixes = [
            "622202", "622203", "622204", "622205", "622206",
            "622207", "622208", "622209", "622210"
        ]
        bin_prefix = np.random.choice(bin_prefixes)
        account = ''.join([str(np.random.randint(0, 10)) for _ in range(10)])
        check_digit = str(np.random.randint(0, 10))
        return bin_prefix + account + check_digit

    else:
        return "未知"


# ==================== 数据记录生成 ====================

def _generate_record(idx: int, source: str, base_time: datetime) -> dict:
    """
    生成单条数据记录

    参数:
        idx: 记录序号
        source: 数据来源类型
        base_time: 基准时间

    返回:
        包含完整字段的数据字典
    """
    record = {}
    record["id"] = f"DY-{source[:2]}-{idx:06d}"
    record["data_source"] = source
    record["timestamp"] = base_time + timedelta(
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60)
    )

    tags = DEYANG_INDUSTRY_TAGS[source]
    record["industry_tag"] = np.random.choice(tags)

    quality_score = max(0, min(100, round(np.random.normal(85, 10), 1)))
    record["quality_score"] = quality_score

    data_size = round(np.random.exponential(scale=50) + 1, 2)
    record["data_size_mb"] = data_size

    has_pii = np.random.random() < 0.55
    if has_pii:
        pii_type = np.random.choice(PII_TYPES)
        record["has_pii"] = True
        record["pii_type"] = pii_type
        record["pii_value"] = _gen_pii(pii_type)
    else:
        record["has_pii"] = False
        record["pii_type"] = ""
        record["pii_value"] = ""

    record["copyright_status"] = np.random.choice(
        ["自有原创", "授权使用", "开源协议", "第三方购买", "爬虫采集"],
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )

    if np.random.random() < 0.05:
        anomaly_type = np.random.choice(["缺失值", "格式错误", "逻辑矛盾", "重复记录", "越界数值"])
        record["anomaly_flag"] = True
        record["anomaly_type"] = anomaly_type
        record["quality_score"] = max(0, quality_score - np.random.randint(20, 61))
    else:
        record["anomaly_flag"] = False
        record["anomaly_type"] = ""

    record["lifecycle_stage"] = np.random.choice(
        ["原始采集", "清洗加工", "融合汇聚", "分析应用", "归档销毁"],
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )

    if source == "研发设计图纸":
        record["cross_domain"] = "成都→德阳"
    elif source in ["生产工艺参数", "供应链物料清单"]:
        record["cross_domain"] = "德阳本地"
    elif source == "售后服务记录":
        record["cross_domain"] = "德阳→全国"
    else:
        record["cross_domain"] = "德阳本地"

    return record


# ==================== 批量生成 ====================

def generate_dataset(n_records: int = 3000, semester: str = "2025春", seed: int = 42) -> pd.DataFrame:
    """
    批量生成合成数据集

    参数:
        n_records: 生成记录数（默认3000）
        semester: 学期标识
        seed: 随机种子

    返回:
        包含所有记录的DataFrame
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    base_time = datetime(2025, 3, 1)

    source_probs = [0.25, 0.2, 0.2, 0.2, 0.15]
    sources = np.random.choice(DATA_SOURCES, size=n_records, p=source_probs)

    for i in range(n_records):
        record = _generate_record(i + 1, sources[i], base_time)
        record["semester"] = semester
        records.append(record)

    df = pd.DataFrame(records)
    df["dataset_version"] = "V4.2"
    df["generated_at"] = datetime.now().isoformat()
    return df


# ==================== 多学期生成 ====================

def generate_multi_semester(semesters: list = None,
                            records_per_semester: int = 7333) -> pd.DataFrame:
    """
    生成多学期合成数据集

    参数:
        semesters: 学期列表（默认2025春、2025秋、2026春）
        records_per_semester: 每学期记录数

    返回:
        合并后的DataFrame
    """
    if semesters is None:
        semesters = ["2025春", "2025秋", "2026春"]

    all_dfs = []
    for i, sem in enumerate(semesters):
        seed = 42 + i * 7
        df = generate_dataset(n_records=records_per_semester, semester=sem, seed=seed)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


# ==================== 高风险数据集生成 ====================

def generate_high_risk_dataset(n_records: int = 1000, seed: int = 666) -> pd.DataFrame:
    """
    生成高风险数据集（用于测试规则引擎硬拦截）
    特征：大量PII、爬虫版权、低质量分数

    参数:
        n_records: 生成记录数
        seed: 随机种子

    返回:
        高风险数据集DataFrame
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    base_time = datetime(2025, 6, 1)

    for i in range(n_records):
        source = np.random.choice(DATA_SOURCES)
        record = _generate_record(i + 1, source, base_time)

        record["has_pii"] = np.random.random() < 0.90
        if record["has_pii"]:
            record["pii_type"] = np.random.choice(PII_TYPES)
            record["pii_value"] = _gen_pii(record["pii_type"])

        if np.random.random() < 0.30:
            record["copyright_status"] = "爬虫采集"

        record["quality_score"] = max(0, min(60, round(np.random.normal(40, 15), 1)))

        if np.random.random() < 0.15:
            record["anomaly_flag"] = True
            record["anomaly_type"] = np.random.choice(["缺失值", "格式错误", "逻辑矛盾"])
            record["quality_score"] = max(0, record["quality_score"] - 20)

        record["semester"] = "高风险测试集"
        record["dataset_version"] = "V4.2_HR"
        records.append(record)

    df = pd.DataFrame(records)
    df["generated_at"] = datetime.now().isoformat()
    return df


# ==================== 向后兼容：异常注入函数 ====================

def inject_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.03, seed: int = 42):
    """
    向数据集中注入异常（向后兼容函数）

    参数:
        df: 原始DataFrame
        anomaly_rate: 异常注入比例（默认3%）
        seed: 随机种子（默认42，保证可复现）

    返回:
        兼容新旧两种调用方式：
        - 旧版：返回 (df, orders) 元组，orders 为异常工单列表
        - 调用方解包为单个 df 时同样可用
    """
    np.random.seed(seed)
    df_out = df.copy()
    orders = []
    n = int(len(df_out) * anomaly_rate)

    if n > 0:
        indices = np.random.choice(df_out.index, n, replace=False)
        types = ["缺失值", "格式错误", "逻辑矛盾", "重复记录", "越界数值"]

        for idx in indices:
            at = np.random.choice(types)
            df_out.loc[idx, "anomaly_flag"] = True
            df_out.loc[idx, "anomaly_type"] = at
            q = df_out.loc[idx, "quality_score"]
            df_out.loc[idx, "quality_score"] = max(0, q - np.random.randint(10, 41))
            orders.append({
                "index": int(idx),
                "anomaly_type": at,
                "description": f"记录 {idx} 检测到 {at}",
                "severity": "中",
            })

    return df_out, orders


# ==================== 主入口 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("成德高端能源装备企业跨域数据流转 - 合成数据生成器 V4.2")
    print("=" * 70)

    print("\n📊 生成常规数据集...")
    normal_df = generate_multi_semester(
        semesters=["2025春", "2025秋", "2026春"],
        records_per_semester=7000
    )
    print(f"✅ 常规数据：{len(normal_df)} 条")
    print(f"   数据源分布：\n{normal_df['data_source'].value_counts()}")

    print("\n⚠️  生成高风险测试集...")
    high_risk_df = generate_high_risk_dataset(n_records=1000)
    print(f"✅ 高风险数据：{len(high_risk_df)} 条")
    print(f"   爬虫采集占比：{(high_risk_df['copyright_status']=='爬虫采集').mean():.1%}")
    print(f"   平均质量分：{high_risk_df['quality_score'].mean():.1f}")

    output_dir = "synthetic_data"
    os.makedirs(output_dir, exist_ok=True)

    normal_path = os.path.join(output_dir, "deyang_normal_data_v4.2.csv")
    high_risk_path = os.path.join(output_dir, "deyang_high_risk_data_v4.2.csv")

    normal_df.to_csv(normal_path, index=False, encoding="utf-8-sig")
    high_risk_df.to_csv(high_risk_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 保存路径：")
    print(f"   常规数据：{os.path.abspath(normal_path)}")
    print(f"   高风险数据：{os.path.abspath(high_risk_path)}")

    print("\n📋 数据摘要：")
    print(f"   总记录数：{len(normal_df) + len(high_risk_df)}")
    print(f"   总字段数：{len(normal_df.columns)}")
    print(f"   含PII比例：{normal_df['has_pii'].mean():.1%}")
    print(f"   异常率：{normal_df['anomaly_flag'].mean():.1%}")
    print(f"   平均质量分：{normal_df['quality_score'].mean():.1f}")
    print(f"   跨域流转比例：{(normal_df['cross_domain']!='德阳本地').mean():.1%}")

    print("\n✅ 数据生成完成！")
