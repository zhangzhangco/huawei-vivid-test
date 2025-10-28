#!/usr/bin/env python3
"""
HDR色调映射专利可视化工具 - Hugging Face Spaces 版本
"""

import sys
import os
import warnings
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 忽略警告
warnings.filterwarnings('ignore')

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(current_dir, 'src'))

def create_hdr_app():
    """创建HDR色调映射应用"""
    
    def phoenix_curve_demo(p, a, dt_low, dt_high):
        """Phoenix曲线演示"""
        try:
            # 输入亮度范围
            L_in = np.linspace(0, 1, 200)
            
            # Phoenix曲线计算
            p_safe = max(p, 0.1)
            L_phoenix = np.power(L_in + 1e-8, 1/p_safe)
            L_out = L_phoenix * a + L_in * (1 - a)
            L_out = L_out * (dt_high - dt_low) + dt_low
            L_out = np.clip(L_out, 0, 1)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 主曲线
            ax1.plot(L_in, L_in, 'k--', alpha=0.5, label='恒等线')
            ax1.plot(L_in, L_out, 'b-', linewidth=2, label='Phoenix曲线')
            ax1.set_xlabel('输入亮度')
            ax1.set_ylabel('输出亮度')
            ax1.set_title('HDR色调映射曲线')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            
            # 差值分析
            diff = L_out - L_in
            ax2.plot(L_in, diff, 'r-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('输入亮度')
            ax2.set_ylabel('映射差值')
            ax2.set_title('色调映射效果')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 质量指标
            contrast = np.std(L_out) / (np.std(L_in) + 1e-8)
            brightness = np.mean(L_out) - np.mean(L_in)
            
            metrics = f"""
**质量指标:**
- 对比度增强: {contrast:.3f}
- 亮度偏移: {brightness:+.3f}
- 动态范围: [{dt_low:.2f}, {dt_high:.2f}]
- 参数: p={p:.2f}, a={a:.2f}
            """
            
            return fig, metrics
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'错误: {str(e)}', ha='center', va='center')
            return fig, f"计算错误: {str(e)}"
    
    def pq_demo(pq_input):
        """PQ转换演示"""
        try:
            # PQ参数
            m1, m2 = 2610.0/16384.0, 2523.0/4096.0*128.0
            c1, c2, c3 = 3424.0/4096.0, 2413.0/4096.0*32.0, 2392.0/4096.0*32.0
            
            # PQ转换
            pq_input = np.clip(pq_input, 0, 1)
            Y_p = np.power(pq_input, 1.0/m2)
            Y_p = np.maximum(Y_p - c1, 0) / (c2 - c3 * Y_p)
            Y_linear = np.power(Y_p, 1.0/m1)
            
            # 可视化
            fig, ax = plt.subplots(figsize=(8, 6))
            pq_range = np.linspace(0, 1, 100)
            Y_p_range = np.power(pq_range, 1.0/m2)
            Y_p_range = np.maximum(Y_p_range - c1, 0) / (c2 - c3 * Y_p_range)
            Y_linear_range = np.power(Y_p_range, 1.0/m1)
            
            ax.plot(pq_range, Y_linear_range, 'g-', linewidth=2, label='PQ EOTF')
            ax.axvline(x=pq_input, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=Y_linear, color='r', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('PQ编码值')
            ax.set_ylabel('线性亮度')
            ax.set_title('PQ转换曲线')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            return fig, f"PQ转换: {pq_input:.3f} → {Y_linear:.6f}"
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'PQ错误: {str(e)}', ha='center', va='center')
            return fig, f"错误: {str(e)}"
    
    # 创建界面
    with gr.Blocks(title="HDR色调映射工具", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🎨 HDR色调映射专利可视化工具
        
        基于Phoenix曲线的HDR色调映射算法可视化系统
        
        ## 功能特性
        - 🎛️ 实时Phoenix曲线可视化
        - 📊 质量指标分析  
        - 🔄 PQ转换演示
        """)
        
        with gr.Tabs():
            with gr.TabItem("Phoenix曲线"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 参数控制")
                        p_slider = gr.Slider(0.5, 4.0, 2.0, step=0.1, label="亮度控制因子 p")
                        a_slider = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="缩放因子 a")
                        dt_low = gr.Slider(0.0, 0.5, 0.0, step=0.01, label="动态范围下限")
                        dt_high = gr.Slider(0.5, 1.0, 1.0, step=0.01, label="动态范围上限")
                        
                    with gr.Column(scale=2):
                        phoenix_plot = gr.Plot()
                        phoenix_metrics = gr.Markdown("调整参数查看效果...")
                
                # 绑定事件
                inputs = [p_slider, a_slider, dt_low, dt_high]
                outputs = [phoenix_plot, phoenix_metrics]
                
                for inp in inputs:
                    inp.change(phoenix_curve_demo, inputs, outputs)
            
            with gr.TabItem("PQ转换"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### PQ转换")
                        pq_input = gr.Slider(0.0, 1.0, 0.5, step=0.001, label="PQ编码值")
                        
                    with gr.Column(scale=2):
                        pq_plot = gr.Plot()
                        pq_result = gr.Markdown("调整PQ值查看转换...")
                
                pq_input.change(pq_demo, [pq_input], [pq_plot, pq_result])
        
        # 初始化
        demo.load(phoenix_curve_demo, inputs, outputs)
        demo.load(pq_demo, [pq_input], [pq_plot, pq_result])
    
    return demo

# 创建应用
print("🚀 启动HDR色调映射工具...")

try:
    # 尝试导入完整版本
    from gradio_app import GradioInterface
    print("✅ 导入完整版本成功")
    interface = GradioInterface()
    app = interface.create_interface()
    print("✅ 完整应用创建成功")
    
except Exception as e:
    print(f"⚠️ 完整版本失败: {e}")
    print("� 使用简指化版本...")
    app = create_hdr_app()
    print("✅ 简化应用创建成功")

print(f"📱 应用类型: {type(app)}")

# Hugging Face Spaces 会自动识别这个变量
if __name__ == "__main__":
    app.launch()