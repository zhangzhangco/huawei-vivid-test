#!/usr/bin/env python3
"""
Auto模式参数估算演示
展示基于图像统计的Phoenix曲线参数自动估算功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.core import (
    AutoModeParameterEstimator, 
    AutoModeInterface, 
    AutoModeConfig,
    ImageStats,
    PhoenixCurveCalculator
)


def create_synthetic_image_stats(scenario: str) -> ImageStats:
    """创建合成图像统计数据用于演示
    
    Args:
        scenario: 场景类型 ("low_light", "high_contrast", "normal", "overexposed")
        
    Returns:
        合成的图像统计信息
    """
    if scenario == "low_light":
        # 低光场景：整体偏暗，动态范围小
        return ImageStats(
            min_pq=0.01,
            max_pq=0.25,
            avg_pq=0.08,
            var_pq=0.003,
            input_format="synthetic",
            processing_path="Low Light Scenario",
            pixel_count=1000000
        )
    elif scenario == "high_contrast":
        # 高对比度场景：动态范围大，方差大
        return ImageStats(
            min_pq=0.02,
            max_pq=0.95,
            avg_pq=0.35,
            var_pq=0.15,
            input_format="synthetic",
            processing_path="High Contrast Scenario",
            pixel_count=1000000
        )
    elif scenario == "overexposed":
        # 过曝场景：整体偏亮
        return ImageStats(
            min_pq=0.3,
            max_pq=0.98,
            avg_pq=0.75,
            var_pq=0.08,
            input_format="synthetic",
            processing_path="Overexposed Scenario",
            pixel_count=1000000
        )
    else:  # normal
        # 正常场景：均衡的亮度分布
        return ImageStats(
            min_pq=0.05,
            max_pq=0.85,
            avg_pq=0.45,
            var_pq=0.06,
            input_format="synthetic",
            processing_path="Normal Scenario",
            pixel_count=1000000
        )


def demonstrate_basic_estimation():
    """演示基本的参数估算功能"""
    print("=== Auto模式参数估算基本演示 ===\n")
    
    # 创建估算器
    estimator = AutoModeParameterEstimator()
    interface = AutoModeInterface(estimator)
    
    # 测试不同场景
    scenarios = ["low_light", "normal", "high_contrast", "overexposed"]
    
    for scenario in scenarios:
        print(f"场景: {scenario.replace('_', ' ').title()}")
        print("-" * 40)
        
        # 创建合成统计数据
        stats = create_synthetic_image_stats(scenario)
        
        # 进行参数估算
        result = estimator.estimate_parameters(stats)
        
        # 显示结果
        print(f"图像统计:")
        print(f"  Min PQ: {stats.min_pq:.3f}")
        print(f"  Max PQ: {stats.max_pq:.3f}")
        print(f"  Avg PQ: {stats.avg_pq:.3f}")
        print(f"  Var PQ: {stats.var_pq:.6f}")
        print()
        
        print(f"估算过程:")
        print(f"  原始估算: p = {result.p_raw:.3f}, a = {result.a_raw:.3f}")
        print(f"  最终参数: p = {result.p_estimated:.3f}, a = {result.a_estimated:.3f}")
        print(f"  参数裁剪: p={result.p_clipped}, a={result.a_clipped}")
        print()
        
        print(f"质量评估:")
        print(f"  置信度: {result.confidence_score:.2f}")
        print(f"  估算质量: {result.estimation_quality}")
        print()
        print("=" * 50)
        print()


def demonstrate_hyperparameter_tuning():
    """演示超参数调节功能"""
    print("=== 超参数调节演示 ===\n")
    
    # 创建估算器和界面
    estimator = AutoModeParameterEstimator()
    interface = AutoModeInterface(estimator)
    
    # 使用正常场景的统计数据
    stats = create_synthetic_image_stats("normal")
    
    print("默认超参数估算:")
    result1 = estimator.estimate_parameters(stats)
    print(f"  超参数: p0={estimator.config.p0}, a0={estimator.config.a0}, α={estimator.config.alpha}, β={estimator.config.beta}")
    print(f"  估算结果: p={result1.p_estimated:.3f}, a={result1.a_estimated:.3f}")
    print()
    
    # 调整超参数
    print("调整超参数后:")
    success, msg = interface.update_hyperparameters(p0=1.5, a0=0.4, alpha=0.8, beta=0.5)
    if success:
        result2 = estimator.estimate_parameters(stats)
        print(f"  超参数: p0={estimator.config.p0}, a0={estimator.config.a0}, α={estimator.config.alpha}, β={estimator.config.beta}")
        print(f"  估算结果: p={result2.p_estimated:.3f}, a={result2.a_estimated:.3f}")
        print(f"  参数变化: Δp={result2.p_estimated-result1.p_estimated:.3f}, Δa={result2.a_estimated-result1.a_estimated:.3f}")
    else:
        print(f"  更新失败: {msg}")
    print()
    
    # 恢复默认设置
    print("恢复默认设置:")
    hyperparams, msg = interface.restore_default_hyperparameters()
    result3 = estimator.estimate_parameters(stats)
    print(f"  {msg}")
    print(f"  估算结果: p={result3.p_estimated:.3f}, a={result3.a_estimated:.3f}")
    print()


def demonstrate_curve_comparison():
    """演示不同估算参数的曲线对比"""
    print("=== 曲线对比演示 ===\n")
    
    # 创建Phoenix曲线计算器
    phoenix_calc = PhoenixCurveCalculator()
    
    # 创建估算器
    estimator = AutoModeParameterEstimator()
    
    # 准备绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Auto模式参数估算 - 不同场景的曲线对比', fontsize=14)
    
    scenarios = ["low_light", "normal", "high_contrast", "overexposed"]
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        ax = axes[i // 2, i % 2]
        
        # 获取统计数据和估算结果
        stats = create_synthetic_image_stats(scenario)
        result = estimator.estimate_parameters(stats)
        
        # 计算曲线
        L_in, L_out = phoenix_calc.get_display_curve(result.p_estimated, result.a_estimated)
        
        # 绘制曲线
        ax.plot(L_in, L_out, color=color, linewidth=2, 
                label=f'Phoenix (p={result.p_estimated:.2f}, a={result.a_estimated:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Identity')
        
        ax.set_title(f'{scenario.replace("_", " ").title()}\n'
                    f'置信度: {result.confidence_score:.2f}, 质量: {result.estimation_quality}')
        ax.set_xlabel('输入亮度 (PQ)')
        ax.set_ylabel('输出亮度 (PQ)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('auto_mode_curves_demo.png', dpi=150, bbox_inches='tight')
    print("曲线对比图已保存为 'auto_mode_curves_demo.png'")
    plt.show()


def demonstrate_interface_features():
    """演示界面功能"""
    print("=== 界面功能演示 ===\n")
    
    # 创建估算器和界面
    estimator = AutoModeParameterEstimator()
    interface = AutoModeInterface(estimator)
    
    # 使用高对比度场景
    stats = create_synthetic_image_stats("high_contrast")
    result = estimator.estimate_parameters(stats)
    
    print("1. 可观测数据:")
    observable_data = interface.get_observable_data()
    print(f"   状态: {observable_data['status']}")
    print(f"   超参数: {observable_data['hyperparameters']}")
    print(f"   图像统计: {observable_data['image_statistics']}")
    print(f"   估算结果: {observable_data['estimation_results']}")
    print(f"   质量评估: {observable_data['quality_assessment']}")
    print()
    
    print("2. 一键应用功能:")
    success, params, msg = interface.apply_estimated_parameters()
    if success:
        print(f"   {msg}")
        print(f"   应用的参数: {params}")
    else:
        print(f"   应用失败: {msg}")
    print()
    
    print("3. 格式化报告:")
    report = interface.format_estimation_report()
    print(report)
    print()
    
    print("4. 超参数范围:")
    ranges = interface.get_hyperparameter_ranges()
    for param, (min_val, max_val) in ranges.items():
        print(f"   {param}: [{min_val}, {max_val}]")


def demonstrate_edge_cases():
    """演示边界情况处理"""
    print("=== 边界情况演示 ===\n")
    
    estimator = AutoModeParameterEstimator()
    interface = AutoModeInterface(estimator)
    
    # 测试极端统计值
    extreme_cases = [
        ("极暗图像", ImageStats(0.001, 0.01, 0.005, 0.0001, "synthetic", "Extreme Dark", 1000000)),
        ("极亮图像", ImageStats(0.95, 0.999, 0.98, 0.0002, "synthetic", "Extreme Bright", 1000000)),
        ("零方差图像", ImageStats(0.5, 0.5, 0.5, 0.0, "synthetic", "Zero Variance", 1000000)),
        ("高方差图像", ImageStats(0.0, 1.0, 0.5, 0.25, "synthetic", "High Variance", 1000000))
    ]
    
    for case_name, stats in extreme_cases:
        print(f"测试: {case_name}")
        print(f"  统计: min={stats.min_pq:.3f}, max={stats.max_pq:.3f}, avg={stats.avg_pq:.3f}, var={stats.var_pq:.6f}")
        
        try:
            result = estimator.estimate_parameters(stats)
            print(f"  估算: p={result.p_estimated:.3f}, a={result.a_estimated:.3f}")
            print(f"  置信度: {result.confidence_score:.2f}, 质量: {result.estimation_quality}")
            print(f"  裁剪: p={result.p_clipped}, a={result.a_clipped}")
        except Exception as e:
            print(f"  错误: {e}")
        print()
    
    # 测试无效超参数
    print("测试无效超参数:")
    invalid_params = [
        ("p0过大", {"p0": 10.0, "a0": 0.3, "alpha": 0.5, "beta": 0.3}),
        ("a0过大", {"p0": 1.0, "a0": 2.0, "alpha": 0.5, "beta": 0.3}),
        ("alpha过大", {"p0": 1.0, "a0": 0.3, "alpha": 5.0, "beta": 0.3}),
        ("beta过大", {"p0": 1.0, "a0": 0.3, "alpha": 0.5, "beta": 2.0})
    ]
    
    for case_name, params in invalid_params:
        print(f"  {case_name}: ", end="")
        success, msg = interface.update_hyperparameters(**params)
        print(f"{'成功' if success else '失败'} - {msg}")


def main():
    """主演示函数"""
    print("Auto模式参数估算功能演示")
    print("=" * 60)
    print()
    
    try:
        # 基本功能演示
        demonstrate_basic_estimation()
        
        # 超参数调节演示
        demonstrate_hyperparameter_tuning()
        
        # 界面功能演示
        demonstrate_interface_features()
        
        # 边界情况演示
        demonstrate_edge_cases()
        
        # 曲线对比演示
        demonstrate_curve_comparison()
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()