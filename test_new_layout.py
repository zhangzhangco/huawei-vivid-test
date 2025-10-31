#!/usr/bin/env python3
"""
测试新的三栏四区布局
"""

import sys
sys.path.insert(0, 'src')

from gradio_app import create_app

if __name__ == "__main__":
    print("=" * 60)
    print("启动HDR色调映射工具 - 三栏四区优化布局")
    print("=" * 60)
    print()
    print("布局说明：")
    print("  左栏 (25%): 参数控制区 - 工作模式、Phoenix参数、样条曲线")
    print("  中栏 (45%): 曲线与指标区 - 曲线图、质量评估、PQ直方图")
    print("  右栏 (30%): 图像与统计区 - 图像对比、统计信息、导出")
    print()
    print("优化特性：")
    print("  ✓ 一屏可见所有核心指标")
    print("  ✓ 调参即时反馈，无需滚动")
    print("  ✓ 折叠面板减少视觉干扰")
    print("  ✓ 语义化颜色提示")
    print()
    
    try:
        app = create_app()
        app.queue()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
