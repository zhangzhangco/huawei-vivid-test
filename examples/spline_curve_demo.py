#!/usr/bin/env python3
"""
样条曲线功能演示
展示PCHIP样条插值、C¹连续性验证和与Phoenix曲线的混合
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from core import (
    PhoenixCurveCalculator, 
    SplineCurveCalculator, 
    SafeCalculator,
    SplineVisualizationHelper
)


def demo_spline_functionality():
    """演示样条曲线功能"""
    print("=== HDR色调映射样条曲线功能演示 ===\n")
    
    # 初始化计算器
    phoenix_calc = PhoenixCurveCalculator()
    spline_calc = SplineCurveCalculator()
    safe_calc = SafeCalculator()
    
    # 设置参数
    p, a = 2.0, 0.5
    th_nodes = [0.2, 0.5, 0.8]
    th_strength = 0.6
    
    print(f"Phoenix参数: p={p}, a={a}")
    print(f"样条节点: {th_nodes}")
    print(f"样条强度: {th_strength}\n")
    
    # 生成输入数据
    x = np.linspace(0, 1, 512)
    
    # 计算Phoenix曲线
    phoenix_curve = phoenix_calc.compute_phoenix_curve(x, p, a)
    print(f"Phoenix曲线计算完成，单调性: {phoenix_calc.validate_monotonicity(phoenix_curve)}")
    
    # 测试样条节点验证
    corrected_nodes, warning = spline_calc.validate_and_correct_nodes(th_nodes)
    print(f"节点验证: {corrected_nodes}")
    if warning:
        print(f"警告: {warning}")
    
    # 创建样条曲线
    spline_curve, spline_success, spline_msg = spline_calc.create_spline_from_phoenix(
        phoenix_curve, x, corrected_nodes
    )
    print(f"样条创建: {'成功' if spline_success else '失败'} - {spline_msg}")
    
    # 验证C¹连续性
    full_nodes = np.array([0.0] + corrected_nodes + [1.0])
    y_nodes = np.interp(full_nodes, x, phoenix_curve)
    is_continuous, continuity_error = spline_calc.verify_c1_continuity(full_nodes, y_nodes)
    print(f"C¹连续性: {'通过' if is_continuous else '失败'}, 最大误差: {continuity_error:.6f}")
    
    # 混合曲线
    blended_curve = spline_calc.blend_with_phoenix(phoenix_curve, spline_curve, th_strength)
    is_monotonic = spline_calc.check_monotonicity(blended_curve)
    print(f"混合曲线单调性: {'通过' if is_monotonic else '失败'}")
    
    # 使用安全计算器进行完整计算
    print("\n=== 安全计算器测试 ===")
    x_safe, final_curve, phoenix_ok, spline_ok, status = safe_calc.safe_combined_curve_calculation(
        p, a, th_nodes, th_strength
    )
    print(f"组合计算: Phoenix={'成功' if phoenix_ok else '失败'}, 样条={'成功' if spline_ok else '失败'}")
    print(f"状态: {status}")
    
    # 生成可视化数据
    viz_data = SplineVisualizationHelper.generate_comparison_data(
        phoenix_curve, spline_curve, x, corrected_nodes
    )
    
    # 计算统计信息
    stats = SplineVisualizationHelper.compute_spline_statistics(phoenix_curve, spline_curve)
    print(f"\n=== 样条统计 ===")
    print(f"最大偏差: {stats['max_deviation']:.6f}")
    print(f"平均偏差: {stats['mean_deviation']:.6f}")
    print(f"RMS偏差: {stats['rms_deviation']:.6f}")
    print(f"正偏差比例: {stats['positive_deviation_ratio']:.3f}")
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    # 主曲线对比
    plt.subplot(2, 2, 1)
    plt.plot(x, x, 'k--', alpha=0.5, label='恒等线')
    plt.plot(x, phoenix_curve, 'b-', linewidth=2, label='Phoenix曲线')
    plt.plot(x, spline_curve, 'r-', linewidth=2, label='样条曲线')
    plt.plot(x, blended_curve, 'g-', linewidth=2, label=f'混合曲线 (强度={th_strength})')
    
    # 标记节点
    node_y_phoenix = np.interp(corrected_nodes, x, phoenix_curve)
    node_y_spline = np.interp(corrected_nodes, x, spline_curve)
    plt.scatter(corrected_nodes, node_y_phoenix, c='blue', s=50, zorder=5)
    plt.scatter(corrected_nodes, node_y_spline, c='red', s=50, zorder=5)
    
    plt.xlabel('输入亮度 (PQ域)')
    plt.ylabel('输出亮度 (PQ域)')
    plt.title('曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 偏差分析
    plt.subplot(2, 2, 2)
    plt.plot(x, viz_data['difference'], 'purple', linewidth=2, label='样条-Phoenix偏差')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('输入亮度 (PQ域)')
    plt.ylabel('偏差')
    plt.title('样条曲线偏差分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 导数分析 (近似)
    plt.subplot(2, 2, 3)
    phoenix_deriv = np.gradient(phoenix_curve, x)
    spline_deriv = np.gradient(spline_curve, x)
    plt.plot(x, phoenix_deriv, 'b-', label='Phoenix导数')
    plt.plot(x, spline_deriv, 'r-', label='样条导数')
    plt.xlabel('输入亮度 (PQ域)')
    plt.ylabel('导数')
    plt.title('导数对比 (数值近似)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 单调性检查可视化
    plt.subplot(2, 2, 4)
    phoenix_diff = np.diff(phoenix_curve)
    spline_diff = np.diff(spline_curve)
    blended_diff = np.diff(blended_curve)
    
    x_diff = x[1:]
    plt.plot(x_diff, phoenix_diff, 'b-', alpha=0.7, label='Phoenix差分')
    plt.plot(x_diff, spline_diff, 'r-', alpha=0.7, label='样条差分')
    plt.plot(x_diff, blended_diff, 'g-', alpha=0.7, label='混合差分')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('输入亮度 (PQ域)')
    plt.ylabel('差分值')
    plt.title('单调性检查 (差分 ≥ 0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spline_curves_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: spline_curves_demo.png")
    
    # 测试边界情况
    print(f"\n=== 边界情况测试 ===")
    
    # 测试强度为0
    zero_strength_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
        phoenix_curve, x, th_nodes, 0.0
    )
    print(f"强度=0: 使用样条={used_spline}, 状态={status}")
    
    # 测试非法节点
    bad_nodes = [0.5, 0.5, 0.5]  # 重复节点
    bad_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
        phoenix_curve, x, bad_nodes, 0.5
    )
    print(f"重复节点: 使用样条={used_spline}, 状态={status}")
    
    # 测试极端强度
    extreme_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
        phoenix_curve, x, th_nodes, 1.0
    )
    print(f"强度=1.0: 使用样条={used_spline}, 状态={status}")
    
    print(f"\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_spline_functionality()