"""
AI企业数据资产化与合规治理虚拟仿真实验
Streamlit Cloud 云端入口
"""

import sys
import os

# 将 V4.5_build 加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V4.5_build"))

# 调用主程序
from main_app import main

if __name__ == "__main__":
    main()
