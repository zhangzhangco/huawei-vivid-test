#!/usr/bin/env python3
"""
时域平滑处理器演示脚本
展示时域平滑算法的工作原理和效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.core import TemporalSmoothingProcessor, QualityMetricsCalculator
import tempfile


def generate_test_sequence(num_frames=20):
    """生成测试参数序列"""
    # 模拟参数变化：基础值 + 噪声 + 趋势
    base_p = 2.0
    base_a = 0.5
    
    # 添加噪声和趋势
    noise_p = np.random.normal(0, 0.3, num_frames)
    noise_a = np.random.normal(0, 0.1, num_frames)
    trend_p = 0.5 * np.sin(np.linspace(0, 4*np.pi, num_frames))
    trend_a = 0.2 * np.cos(np.linspace(0, 2*np.pi, num_frames))
    
    params_sequence = []
    distortions = []
    
    for i in range(num_frames):
        p = base_p + trend_p[i] + noise_p[i]
        a = base_a + trend_a[i] + noise_a[i]
        
        # 确保参数在有效范围内
        p = np.clip(p, 0.1, 6.0)
        a = np.clip(a, 0.0, 1.0)
        
        params_sequence.append({'p': p, 'a': a})
        
        # 模拟失真计算
        distortion = abs(p - base_p) * 0.1 + abs(a - base_a) * 0.2 + np.random.uniform(0.01, 0.05)
        distortions.append(distortion)
        
    return params_sequence, distortions


def demonstrate_temporal_smoothing():
    """演示时域平滑效果"""
    print("=== HDR时域平滑处理器演示 ===\n")
    
    # 创建临时文件用于状态存储
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
    
    try:
        # 初始化处理器
        processor = TemporalSmoothingProcessor(window_size=9, temporal_file=temp_file)
        print(f"初始化时域平滑处理器:")
        print(f"- 窗口大小: {processor.window_size}")
        print(f"- 平滑强度范围: {processor.lambda_range}")
        print(f"- 状态文件: {temp_file}\n")
        
        # 生成测试序列
        print("生成测试参数序列...")
        params_sequence, distortions = generate_test_sequence(25)
        print(f"生成了 {len(params_sequence)} 帧参数\n")
        
        # 逐帧处理并记录结果
        raw_p_values = []
        raw_a_values = []
        filtered_p_values = []
        filtered_a_values = []
        frame_numbers = []
        
        lambda_smooth = 0.3
        
        print("开始逐帧处理:")
        print("帧号 | 原始p值 | 滤波p值 | 原始a值 | 滤波a值 | 失真值 | 窗口利用率")
        print("-" * 75)
        
        for i, (params, distortion) in enumerate(zip(params_sequence, distortions)):
            # 添加当前帧参数
            processor.add_frame_parameters(params, distortion)
            
            # 应用时域滤波
            if len(processor.parameter_history) >= 2:
                filtered_params = processor.apply_temporal_filter(params, lambda_smooth)
            else:
                filtered_params = params.copy()
                
            # 记录结果
            raw_p_values.append(params['p'])
            raw_a_values.append(params['a'])
            filtered_p_values.append(filtered_params['p'])
            filtered_a_values.append(filtered_params['a'])
            frame_numbers.append(i + 1)
            
            # 获取状态信息
            state_info = processor.get_state_info()
            
            # 打印进度
            if i < 10 or i % 5 == 0:  # 前10帧和每5帧打印一次
                print(f"{i+1:3d}  | {params['p']:7.3f} | {filtered_params['p']:7.3f} | "
                      f"{params['a']:7.3f} | {filtered_params['a']:7.3f} | "
                      f"{distortion:6.3f} | {state_info['window_utilization']:8.1%}")
                
        print("\n处理完成!\n")
        
        # 获取最终统计信息
        stats = processor.get_smoothing_stats()
        print("=== 平滑效果统计 ===")
        print(f"总帧数: {stats.frame_count}")
        print(f"窗口利用率: {stats.window_utilization:.1%}")
        print(f"p参数原始方差: {stats.p_var_raw:.6f}")
        print(f"p参数滤波方差: {stats.p_var_filtered:.6f}")
        print(f"方差减少率: {stats.variance_reduction:.1%}")
        print(f"最近Δp原始: {stats.delta_p_raw:.6f}")
        print(f"最近Δp滤波: {stats.delta_p_filtered:.6f}")
        
        # 验证平滑效果
        is_effective, msg = processor.validate_smoothing_effectiveness()
        print(f"\n平滑效果验证: {'✓' if is_effective else '✗'} {msg}")
        
        # 可视化结果
        print("\n生成可视化图表...")
        create_visualization(frame_numbers, raw_p_values, filtered_p_values, 
                           raw_a_values, filtered_a_values, distortions, stats)
        
        # 演示状态持久化
        print("\n=== 状态持久化演示 ===")
        export_data = processor.export_temporal_data()
        print(f"导出数据包含 {len(export_data)} 个主要部分:")
        for key in export_data.keys():
            print(f"- {key}")
            
        # 模拟重启后加载状态
        print("\n模拟系统重启后状态恢复...")
        new_processor = TemporalSmoothingProcessor(window_size=9, temporal_file=temp_file)
        print(f"恢复后窗口中有 {len(new_processor.parameter_history)} 帧历史数据")
        print(f"总帧数: {new_processor.frame_count}")
        
        # 演示模拟功能
        print("\n=== 平滑效果模拟 ===")
        test_params = params_sequence[:10]
        test_distortions = distortions[:10]
        
        simulation_result = processor.simulate_smoothing_effect(test_params, test_distortions, 0.4)
        print(f"模拟了 {len(simulation_result['raw_sequence'])} 帧")
        print(f"模拟平滑效果: {'有效' if simulation_result['effectiveness'] else '无效'}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    print("\n演示完成!")


def create_visualization(frame_numbers, raw_p, filtered_p, raw_a, filtered_a, distortions, stats):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HDR时域平滑处理效果', fontsize=16, fontweight='bold')
    
    # p参数对比
    axes[0, 0].plot(frame_numbers, raw_p, 'b-', alpha=0.7, linewidth=1, label='原始p值')
    axes[0, 0].plot(frame_numbers, filtered_p, 'r-', linewidth=2, label='滤波p值')
    axes[0, 0].set_title('p参数时域平滑效果')
    axes[0, 0].set_xlabel('帧号')
    axes[0, 0].set_ylabel('p值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # a参数对比
    axes[0, 1].plot(frame_numbers, raw_a, 'b-', alpha=0.7, linewidth=1, label='原始a值')
    axes[0, 1].plot(frame_numbers, filtered_a, 'r-', linewidth=2, label='滤波a值')
    axes[0, 1].set_title('a参数时域平滑效果')
    axes[0, 1].set_xlabel('帧号')
    axes[0, 1].set_ylabel('a值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 失真变化
    axes[1, 0].plot(frame_numbers, distortions, 'g-', linewidth=1.5, label='感知失真')
    axes[1, 0].set_title('感知失真变化')
    axes[1, 0].set_xlabel('帧号')
    axes[1, 0].set_ylabel('失真值')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 方差对比
    var_data = ['原始p方差', '滤波p方差']
    var_values = [stats.p_var_raw, stats.p_var_filtered]
    colors = ['lightblue', 'lightcoral']
    
    bars = axes[1, 1].bar(var_data, var_values, color=colors, alpha=0.7)
    axes[1, 1].set_title(f'方差减少效果 (减少{stats.variance_reduction:.1%})')
    axes[1, 1].set_ylabel('方差值')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, var_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = 'temporal_smoothing_demo.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存为: {output_file}")
    
    # 显示图表（如果在支持的环境中）
    try:
        plt.show()
    except:
        print("无法显示图表，但已保存到文件")


def demonstrate_integration_example():
    """演示与质量指标的集成使用"""
    print("\n=== 集成使用示例 ===")
    
    # 创建质量指标计算器
    quality_calc = QualityMetricsCalculator()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
    
    try:
        # 创建时域处理器
        temporal_processor = TemporalSmoothingProcessor(window_size=7, temporal_file=temp_file)
        
        print("模拟实际HDR视频处理场景...")
        
        # 模拟连续帧处理
        for frame_idx in range(15):
            # 模拟当前帧的参数估算
            base_p = 2.0 + 0.3 * np.sin(frame_idx * 0.5)
            base_a = 0.5 + 0.1 * np.cos(frame_idx * 0.3)
            
            # 添加随机扰动
            current_params = {
                'p': base_p + np.random.normal(0, 0.2),
                'a': base_a + np.random.normal(0, 0.05)
            }
            
            # 确保参数在有效范围
            current_params['p'] = np.clip(current_params['p'], 0.1, 6.0)
            current_params['a'] = np.clip(current_params['a'], 0.0, 1.0)
            
            # 模拟质量指标计算
            L_in = np.random.rand(1000) * 0.8  # 模拟输入亮度
            L_out = L_in ** current_params['p'] / (L_in ** current_params['p'] + current_params['a'] ** current_params['p'])
            distortion = quality_calc.compute_perceptual_distortion(L_in, L_out)
            
            # 添加到时域处理器
            temporal_processor.add_frame_parameters(current_params, distortion)
            
            # 应用时域平滑
            if frame_idx >= 2:  # 有足够历史后开始平滑
                smoothed_params = temporal_processor.apply_temporal_filter(current_params, 0.35)
                
                print(f"帧 {frame_idx+1:2d}: p={current_params['p']:.3f}→{smoothed_params['p']:.3f}, "
                      f"a={current_params['a']:.3f}→{smoothed_params['a']:.3f}, D'={distortion:.4f}")
            else:
                print(f"帧 {frame_idx+1:2d}: p={current_params['p']:.3f} (无平滑), "
                      f"a={current_params['a']:.3f} (无平滑), D'={distortion:.4f}")
                      
        # 最终统计
        final_stats = temporal_processor.get_smoothing_stats()
        print(f"\n最终统计:")
        print(f"- 处理帧数: {final_stats.frame_count}")
        print(f"- 方差减少: {final_stats.variance_reduction:.1%}")
        print(f"- 窗口利用率: {final_stats.window_utilization:.1%}")
        
        # 验证效果
        is_effective, msg = temporal_processor.validate_smoothing_effectiveness()
        print(f"- 平滑效果: {msg}")
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    # 设置随机种子以获得可复现的结果
    np.random.seed(42)
    
    # 运行主演示
    demonstrate_temporal_smoothing()
    
    # 运行集成示例
    demonstrate_integration_example()