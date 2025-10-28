#!/usr/bin/env python3
"""
错误处理系统演示
展示UI错误处理器、错误恢复系统和边界检查器的功能
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import (
    UIErrorHandler, ErrorRecoverySystem, BoundaryChecker, SafeCalculator,
    ErrorSeverity
)


def demo_ui_error_handler():
    """演示UI错误处理器"""
    print("=" * 60)
    print("UI错误处理器演示")
    print("=" * 60)
    
    error_handler = UIErrorHandler()
    
    # 1. 创建不同类型的错误
    print("\n1. 创建不同严重程度的错误:")
    
    error_handler.add_error(ErrorSeverity.INFO, "系统信息", "系统启动完成", "无需操作")
    error_handler.add_error(ErrorSeverity.WARNING, "参数警告", "参数p=5.5可能导致数值不稳定", "建议调整到4.0以下")
    error_handler.add_error(ErrorSeverity.ERROR, "计算错误", "Phoenix曲线计算失败", "请检查参数设置")
    error_handler.add_error(ErrorSeverity.CRITICAL, "系统错误", "内存不足", "请关闭其他应用程序")
    
    # 2. 显示错误摘要
    print("\n2. 错误摘要:")
    summary = error_handler.get_error_summary()
    print(f"   总错误数: {summary['total_errors']}")
    print(f"   错误计数: {summary['error_count']}")
    print(f"   警告计数: {summary['warning_count']}")
    print(f"   系统状态: {summary['status']}")
    
    # 3. 显示最近错误
    print("\n3. 最近错误:")
    recent_errors = error_handler.get_recent_errors(3)
    for i, error in enumerate(recent_errors, 1):
        formatted = error_handler.format_error_for_display(error)
        print(f"   {i}. {formatted}")
    
    # 4. 图像验证演示
    print("\n4. 图像验证演示:")
    
    # 正常图像
    normal_image = np.random.rand(100, 100, 3).astype(np.float32)
    is_valid, msg = error_handler.validate_image_upload(normal_image)
    print(f"   正常图像 (100x100): {is_valid} - {msg}")
    
    # 过大图像
    large_image = np.random.rand(3000, 3000, 3).astype(np.float32)
    is_valid, msg = error_handler.validate_image_upload(large_image, max_pixels=1_000_000)
    print(f"   大图像 (3000x3000): {is_valid} - {msg}")
    
    # None图像
    is_valid, msg = error_handler.validate_image_upload(None)
    print(f"   空图像: {is_valid} - {msg}")


def demo_boundary_checker():
    """演示边界检查器"""
    print("\n" + "=" * 60)
    print("边界检查器演示")
    print("=" * 60)
    
    boundary_checker = BoundaryChecker()
    
    # 1. 正常参数检查
    print("\n1. 正常参数检查:")
    normal_params = {'p': 2.0, 'a': 0.5, 'dt_low': 0.05, 'dt_high': 0.10}
    is_valid, violations = boundary_checker.check_all_boundaries(normal_params)
    print(f"   参数: {normal_params}")
    print(f"   有效: {is_valid}")
    print(f"   违反条件数: {len(violations)}")
    
    # 2. 异常参数检查
    print("\n2. 异常参数检查:")
    invalid_params = {
        'p': 10.0,  # 超出范围
        'a': 1.5,   # 超出范围
        'dt_low': 0.15,  # 与dt_high冲突
        'dt_high': 0.05,
        'min_display_pq': 0.8,  # 与max_display_pq冲突
        'max_display_pq': 0.2
    }
    is_valid, violations = boundary_checker.check_all_boundaries(invalid_params)
    print(f"   参数: {invalid_params}")
    print(f"   有效: {is_valid}")
    print(f"   违反条件数: {len(violations)}")
    
    # 3. 显示违反条件详情
    print("\n3. 违反条件详情:")
    for i, violation in enumerate(violations[:5], 1):  # 只显示前5个
        print(f"   {i}. {violation.parameter}: {violation.constraint_description}")
        if violation.suggested_value is not None:
            print(f"      建议值: {violation.suggested_value}")
    
    # 4. 自动修正
    print("\n4. 自动修正:")
    corrected_params = boundary_checker.auto_correct_violations(invalid_params, violations)
    print(f"   修正前: p={invalid_params.get('p')}, a={invalid_params.get('a')}")
    print(f"   修正后: p={corrected_params.get('p')}, a={corrected_params.get('a')}")
    
    # 5. 违反条件报告
    print("\n5. 违反条件报告:")
    report = boundary_checker.create_violation_report(violations)
    print(report[:500] + "..." if len(report) > 500 else report)


def demo_error_recovery():
    """演示错误恢复系统"""
    print("\n" + "=" * 60)
    print("错误恢复系统演示")
    print("=" * 60)
    
    recovery_system = ErrorRecoverySystem()
    
    # 1. 保存正常状态
    print("\n1. 保存系统状态:")
    normal_state = {'p': 2.0, 'a': 0.5, 'mode': '艺术模式'}
    state = recovery_system.save_state(normal_state, is_valid=True)
    print(f"   保存状态: {normal_state}")
    print(f"   状态有效: {state.is_valid}")
    
    # 2. 参数验证错误恢复
    print("\n2. 参数验证错误恢复:")
    invalid_params = {'p': 10.0, 'a': 1.5}
    actions = recovery_system.analyze_error("parameter_validation", invalid_params, "参数超出范围")
    print(f"   错误参数: {invalid_params}")
    print(f"   恢复策略数: {len(actions)}")
    
    for i, action in enumerate(actions[:3], 1):  # 显示前3个策略
        print(f"   {i}. {action.description} (成功率: {action.success_probability:.1%})")
    
    # 3. 执行自动恢复
    print("\n3. 执行自动恢复:")
    success, message, recovered_params = recovery_system.auto_recover(
        "parameter_validation", invalid_params, "参数验证失败"
    )
    print(f"   恢复成功: {success}")
    print(f"   恢复消息: {message}")
    print(f"   恢复参数: {recovered_params}")
    
    # 4. 单调性错误恢复
    print("\n4. 单调性错误恢复:")
    mono_params = {'p': 5.5, 'a': 0.1}  # 可能导致非单调
    actions = recovery_system.analyze_error("monotonicity_violation", mono_params, "曲线非单调")
    print(f"   问题参数: {mono_params}")
    print(f"   恢复策略:")
    for i, action in enumerate(actions[:2], 1):
        print(f"   {i}. {action.description}")
    
    # 5. 恢复系统状态
    print("\n5. 恢复系统状态:")
    status = recovery_system.get_recovery_status()
    print(f"   恢复尝试: {status['recovery_attempts']}/{status['max_recovery_attempts']}")
    print(f"   系统稳定: {status['system_stable']}")
    print(f"   状态历史: {status['state_history_length']}个")


def demo_safe_calculator():
    """演示安全计算器"""
    print("\n" + "=" * 60)
    print("安全计算器演示")
    print("=" * 60)
    
    safe_calc = SafeCalculator()
    
    # 1. 正常计算
    print("\n1. 正常Phoenix曲线计算:")
    L = np.linspace(0, 1, 100)
    L_out, success, msg, status = safe_calc.safe_phoenix_calculation_enhanced(L, 2.0, 0.5)
    print(f"   输入: p=2.0, a=0.5")
    print(f"   计算成功: {success}")
    print(f"   状态消息: {msg}")
    print(f"   详细状态: {status}")
    
    # 2. 异常参数计算
    print("\n2. 异常参数计算:")
    L_out, success, msg, status = safe_calc.safe_phoenix_calculation_enhanced(L, 10.0, 1.5)
    print(f"   输入: p=10.0, a=1.5 (超出范围)")
    print(f"   计算成功: {success}")
    print(f"   状态消息: {msg}")
    print(f"   参数验证: {status['parameter_validation']}")
    print(f"   恢复应用: {status['recovery_applied']}")
    
    # 3. 全面参数验证
    print("\n3. 全面参数验证:")
    test_params = {'p': 8.0, 'a': 1.2, 'dt_low': 0.15, 'dt_high': 0.05}
    is_valid, corrected, errors = safe_calc.comprehensive_parameter_validation(test_params)
    print(f"   测试参数: {test_params}")
    print(f"   验证通过: {is_valid}")
    print(f"   错误数量: {len(errors)}")
    print(f"   修正参数: p={corrected.get('p'):.2f}, a={corrected.get('a'):.2f}")
    
    # 4. 系统状态
    print("\n4. 系统状态:")
    system_status = safe_calc.get_comprehensive_system_status()
    print(f"   系统稳定: {system_status['system_stable']}")
    print(f"   错误计数: {system_status['error_count']}")
    print(f"   自动恢复: {system_status['auto_recovery_enabled']}")
    print(f"   最近错误: {system_status['recent_errors']}")
    
    # 5. 诊断报告
    print("\n5. 系统诊断报告:")
    report = safe_calc.create_system_diagnostic_report()
    print(report[:800] + "..." if len(report) > 800 else report)


def demo_error_visualization():
    """演示错误可视化"""
    print("\n" + "=" * 60)
    print("错误可视化演示")
    print("=" * 60)
    
    error_handler = UIErrorHandler()
    
    # 创建错误图表
    print("\n创建错误图表...")
    
    # 1. 错误图表
    error_fig = error_handler.create_error_plot("参数p=10.0超出有效范围[0.1, 6.0]", "curve")
    
    # 2. 警告图表
    warning_fig = error_handler.create_warning_plot("曲线非单调，已自动回退", "curve")
    
    # 保存图表
    try:
        error_fig.savefig('error_plot_demo.png', dpi=100, bbox_inches='tight')
        warning_fig.savefig('warning_plot_demo.png', dpi=100, bbox_inches='tight')
        print("   错误图表已保存: error_plot_demo.png")
        print("   警告图表已保存: warning_plot_demo.png")
    except Exception as e:
        print(f"   图表保存失败: {e}")
    
    plt.close('all')  # 关闭所有图表


def main():
    """主函数"""
    print("HDR色调映射错误处理系统演示")
    print("=" * 60)
    
    try:
        # 运行各个演示
        demo_ui_error_handler()
        demo_boundary_checker()
        demo_error_recovery()
        demo_safe_calculator()
        demo_error_visualization()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()