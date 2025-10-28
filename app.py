#!/usr/bin/env python3
"""
HDR色调映射专利可视化工具 - Hugging Face Spaces 入口文件
"""

import sys
import os
import warnings

# 忽略警告信息，保持日志清洁
warnings.filterwarnings('ignore')

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_spaces_app():
    """为 Spaces 环境创建应用"""
    
    try:
        # 导入主应用
        from gradio_app import create_app
        
        # 创建Gradio应用实例
        print("🚀 启动HDR色调映射专利可视化工具...")
        app = create_app()
        
        print("✅ 应用创建成功！")
        return app
        
    except ImportError as e:
        print(f"⚠️  导入错误: {e}")
        print("🔄 尝试创建简化版本...")
        return create_fallback_app()
        
    except Exception as e:
        print(f"❌ 应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return create_error_app(str(e))

def create_fallback_app():
    """创建简化的备用应用"""
    
    import gradio as gr
    import numpy as np
    import matplotlib.pyplot as plt
    
    def phoenix_demo(p, a):
        """简化的Phoenix曲线演示"""
        try:
            L_in = np.linspace(0, 1, 100)
            L_out = np.power(L_in + 1e-8, 1/max(p, 0.1)) * a + L_in * (1-a)
            L_out = np.clip(L_out, 0, 1)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(L_in, L_in, 'k--', alpha=0.5, label='恒等线')
            ax.plot(L_in, L_out, 'b-', linewidth=2, label=f'Phoenix曲线')
            ax.set_xlabel('输入亮度')
            ax.set_ylabel('输出亮度')
            ax.set_title('HDR色调映射曲线')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            
            return fig
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'错误: {str(e)}', ha='center', va='center')
            return fig
    
    with gr.Blocks(title="HDR色调映射工具") as app:
        gr.Markdown("# HDR色调映射专利可视化工具 - 简化版")
        
        with gr.Row():
            p_slider = gr.Slider(0.5, 4.0, 2.0, label="亮度控制因子 p")
            a_slider = gr.Slider(0.0, 1.0, 0.5, label="缩放因子 a")
        
        plot = gr.Plot()
        
        for slider in [p_slider, a_slider]:
            slider.change(phoenix_demo, [p_slider, a_slider], plot)
        
        app.load(phoenix_demo, [p_slider, a_slider], plot)
    
    return app

def create_error_app(error_msg):
    """创建错误显示应用"""
    
    import gradio as gr
    
    def show_error():
        return f"应用启动失败: {error_msg}\n\n请检查依赖安装或联系开发者。"
    
    app = gr.Interface(
        fn=show_error,
        inputs=[],
        outputs="text",
        title="HDR色调映射工具 - 启动错误"
    )
    
    return app

# 创建应用实例
print("🌟 HDR色调映射专利可视化工具 - Spaces版")
app = create_spaces_app()

# Hugging Face Spaces 会自动识别并启动这个 app 变量