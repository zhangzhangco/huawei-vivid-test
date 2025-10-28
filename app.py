#!/usr/bin/env python3
"""
HDR色调映射专利可视化工具 - Hugging Face Spaces 入口文件
"""

import sys
import os
import warnings
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 确保使用非交互式后端

# 忽略警告信息，保持日志清洁
warnings.filterwarnings('ignore')

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, 'src'))

def create_spaces_app():
    """为 Spaces 环境创建应用"""
    
    try:
        # 尝试导入完整应用
        print("🚀 尝试启动完整版HDR色调映射工具...")
        
        # 检查核心模块是否可用
        try:
            from core.phoenix_calculator import PhoenixCurveCalculator
            from core.pq_converter import PQConverter
            print("✅ 核心模块导入成功")
            
            # 导入主应用
            from gradio_app import GradioInterface
            interface = GradioInterface()
            app = interface.create_interface()
            
            print("✅ 完整应用创建成功！")
            return app
            
        except ImportError as e:
            print(f"⚠️  核心模块导入失败: {e}")
            print("🔄 使用简化版本...")
            return create_simplified_app()
        
    except Exception as e:
        print(f"❌ 完整应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return create_simplified_app()

def create_simplified_app():
    """创建简化但功能完整的应用"""
    
    def phoenix_curve_calculation(p, a, dt_low, dt_high):
        """Phoenix曲线计算和可视化"""
        try:
            # 输入亮度范围
            L_in = np.linspace(0, 1, 200)
            
            # Phoenix曲线计算
            # 避免除零错误
            p_safe = max(p, 0.1)
            
            # 基础Phoenix变换
            L_phoenix = np.power(L_in + 1e-8, 1/p_safe)
            
            # 应用缩放因子
            L_out = L_phoenix * a + L_in * (1 - a)
            
            # 应用动态范围调整
            L_out = L_out * (dt_high - dt_low) + dt_low
            
            # 确保输出在合理范围内
            L_out = np.clip(L_out, 0, 1)
            
            # 创建可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 主曲线图
            ax1.plot(L_in, L_in, 'k--', alpha=0.5, label='恒等线 (y=x)')
            ax1.plot(L_in, L_out, 'b-', linewidth=2, label='Phoenix曲线')
            ax1.set_xlabel('输入亮度')
            ax1.set_ylabel('输出亮度')
            ax1.set_title('HDR色调映射曲线')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            
            # 差值分析图
            diff = L_out - L_in
            ax2.plot(L_in, diff, 'r-', linewidth=2, label='映射差值')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('输入亮度')
            ax2.set_ylabel('输出差值')
            ax2.set_title('色调映射效果分析')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # 计算质量指标
            contrast_enhancement = np.std(L_out) / (np.std(L_in) + 1e-8)
            brightness_shift = np.mean(L_out) - np.mean(L_in)
            
            metrics_text = f"""
            **质量指标分析:**
            - 对比度增强: {contrast_enhancement:.3f}
            - 亮度偏移: {brightness_shift:+.3f}
            - 动态范围: [{dt_low:.2f}, {dt_high:.2f}]
            - 参数设置: p={p:.2f}, a={a:.2f}
            """
            
            return fig, metrics_text
            
        except Exception as e:
            # 错误处理
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'计算错误: {str(e)}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig, f"错误: {str(e)}"
    
    def pq_conversion_demo(L_in_pq):
        """PQ转换演示"""
        try:
            # PQ EOTF 参数 (ITU-R BT.2100)
            m1 = 2610.0 / 16384.0
            m2 = 2523.0 / 4096.0 * 128.0
            c1 = 3424.0 / 4096.0
            c2 = 2413.0 / 4096.0 * 32.0
            c3 = 2392.0 / 4096.0 * 32.0
            
            # PQ到线性转换
            L_in_pq = np.clip(L_in_pq, 0, 1)
            
            # PQ EOTF
            Y_p = np.power(L_in_pq, 1.0/m2)
            Y_p = np.maximum(Y_p - c1, 0) / (c2 - c3 * Y_p)
            Y_linear = np.power(Y_p, 1.0/m1)
            
            # 创建可视化
            fig, ax = plt.subplots(figsize=(8, 6))
            
            pq_range = np.linspace(0, 1, 100)
            Y_p_range = np.power(pq_range, 1.0/m2)
            Y_p_range = np.maximum(Y_p_range - c1, 0) / (c2 - c3 * Y_p_range)
            Y_linear_range = np.power(Y_p_range, 1.0/m1)
            
            ax.plot(pq_range, Y_linear_range, 'g-', linewidth=2, label='PQ EOTF曲线')
            ax.axvline(x=L_in_pq, color='r', linestyle='--', alpha=0.7, label=f'输入值: {L_in_pq:.3f}')
            ax.axhline(y=Y_linear, color='r', linestyle='--', alpha=0.7, label=f'输出值: {Y_linear:.3f}')
            
            ax.set_xlabel('PQ编码值')
            ax.set_ylabel('线性亮度值')
            ax.set_title('PQ (Perceptual Quantizer) 转换曲线')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            return fig, f"PQ转换结果: {L_in_pq:.3f} → {Y_linear:.6f}"
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'PQ转换错误: {str(e)}', ha='center', va='center')
            return fig, f"错误: {str(e)}"
    
    # 创建Gradio界面
    with gr.Blocks(
        title="HDR色调映射专利可视化工具",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🎨 HDR色调映射专利可视化工具
        
        基于Phoenix曲线算法的HDR色调映射可视化系统，支持实时参数调节和质量分析。
        
        ## 功能特性
        - 🎛️ **实时Phoenix曲线可视化**
        - 📊 **质量指标分析**  
        - 🔄 **PQ转换演示**
        - ⚡ **参数实时调节**
        """)
        
        with gr.Tabs():
            # Phoenix曲线标签页
            with gr.TabItem("Phoenix曲线"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 参数控制")
                        p_slider = gr.Slider(0.5, 4.0, 2.0, step=0.1, label="亮度控制因子 p")
                        a_slider = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="缩放因子 a")
                        dt_low_slider = gr.Slider(0.0, 0.5, 0.0, step=0.01, label="动态范围下限")
                        dt_high_slider = gr.Slider(0.5, 1.0, 1.0, step=0.01, label="动态范围上限")
                        
                    with gr.Column(scale=2):
                        phoenix_plot = gr.Plot(label="Phoenix曲线可视化")
                        phoenix_metrics = gr.Markdown("等待参数调整...")
                
                # 绑定事件
                inputs = [p_slider, a_slider, dt_low_slider, dt_high_slider]
                outputs = [phoenix_plot, phoenix_metrics]
                
                for input_component in inputs:
                    input_component.change(phoenix_curve_calculation, inputs, outputs)
            
            # PQ转换标签页
            with gr.TabItem("PQ转换"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### PQ转换参数")
                        pq_input = gr.Slider(0.0, 1.0, 0.5, step=0.001, label="PQ编码输入值")
                        
                    with gr.Column(scale=2):
                        pq_plot = gr.Plot(label="PQ转换曲线")
                        pq_result = gr.Markdown("等待输入...")
                
                pq_input.change(pq_conversion_demo, [pq_input], [pq_plot, pq_result])
        
        # 初始化显示
        app.load(phoenix_curve_calculation, inputs, outputs)
        app.load(pq_conversion_demo, [pq_input], [pq_plot, pq_result])
    
    return app



# 创建应用实例
print("🌟 HDR色调映射专利可视化工具 - Spaces版")

try:
    app = create_spaces_app()
    print(f"✅ 应用创建成功: {type(app)}")
    
except Exception as e:
    print(f"❌ 应用创建失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 创建最基础的错误显示应用
    app = gr.Interface(
        fn=lambda: f"应用启动失败: {str(e)}\n\n请检查依赖或联系开发者。",
        inputs=[],
        outputs="text",
        title="HDR色调映射工具 - 启动错误"
    )

# 确保应用对象存在
if 'app' not in locals() or app is None:
    print("⚠️  创建备用应用...")
    app = gr.Interface(
        fn=lambda: "HDR色调映射工具 - 系统初始化中...",
        inputs=[],
        outputs="text",
        title="HDR色调映射工具"
    )

print(f"🚀 最终应用: {type(app)}")

# Hugging Face Spaces 自动识别此变量