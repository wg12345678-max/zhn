"""
AI企业数据资产化与合规治理虚拟仿真实验
Streamlit Cloud 云端入口
用法：streamlit run streamlit_app.py
"""

import sys
import os
import subprocess

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 V4.5_build 加入模块搜索路径
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "V4.5_build"))

# 直接执行原入口脚本（因为 main_app.py 是顶层脚本，没有 main() 函数）
exec(open(os.path.join(current_dir, "V4.5_build", "main_app.py")).read())
