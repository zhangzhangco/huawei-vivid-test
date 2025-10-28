#!/usr/bin/env python3
"""
HDR色调映射专利可视化工具 - Gradio界面启动器
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from gradio_app import main
    
    if __name__ == "__main__":
        print("=" * 60)
        print("HDR色调映射专利可视化工具")
        print("基于Gradio框架的交互式Web界面")
        print("=" * 60)
        print()
        print("功能特性:")
        print("- 实时Phoenix曲线可视化")
        print("- 参数控制面板（滑块、单选按钮）")
        print("- 图像上传和对比显示")
        print("- 质量指标和模式建议显示")
        print("- 时域平滑演示")
        print("- 样条曲线扩展功能")
        print("- 自动模式参数估算")
        print()
        print("启动中...")
        print("界面将在浏览器中打开: http://localhost:7860")
        print()
        
        main()
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖:")
    print("pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)