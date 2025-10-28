"""
HDR色调映射专利可视化工具主程序
演示核心功能的基本使用
"""

import numpy as np
import matplotlib.pyplot as plt
from core import PQConverter, PhoenixCurveCalculator, SafeCalculator


def demo_phoenix_curve():
    """演示Phoenix曲线计算"""
    print("=== Phoenix曲线演示 ===")
    
    # 创建计算器
    calculator = PhoenixCurveCalculator()
    
    # 获取显示曲线
    L, L_out = calculator.get_display_curve(p=2.0, a=0.5)
    
    print(f"采样点数: {len(L)}")
    print(f"输入范围: [{L[0]:.3f}, {L[-1]:.3f}]")
    print(f"输出范围: [{L_out[0]:.6f}, {L_out[-1]:.6f}]")
    print(f"单调性: {calculator.validate_monotonicity(L_out)}")
    
    # 端点归一化演示
    normalized = calculator.normalize_endpoints(L_out, 0.1, 0.9)
    print(f"归一化后端点: [{normalized[0]:.6f}, {normalized[-1]:.6f}]")
    
    return L, L_out, normalized


def demo_pq_conversion():
    """演示PQ转换"""
    print("\n=== PQ转换演示 ===")
    
    converter = PQConverter()
    
    # 测试不同亮度值
    test_nits = [0, 100, 1000, 4000, 10000]
    
    print("线性光 -> PQ域转换:")
    for nits in test_nits:
        pq = converter.linear_to_pq(nits)
        recovered = converter.pq_to_linear(pq)
        print(f"  {nits:5.0f} nits -> {pq:.6f} PQ -> {recovered:5.0f} nits")
        
    # sRGB转换演示
    print("\nsRGB -> 线性光转换:")
    srgb_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    for srgb in srgb_values:
        linear = converter.srgb_to_linear(srgb)
        recovered = converter.linear_to_srgb(linear)
        print(f"  {srgb:.1f} sRGB -> {linear:.6f} linear -> {recovered:.6f} sRGB")


def demo_safe_calculation():
    """演示安全计算"""
    print("\n=== 安全计算演示 ===")
    
    safe_calc = SafeCalculator()
    
    # 测试有效参数
    L = np.linspace(0, 1, 100)
    result, success, msg = safe_calc.safe_phoenix_calculation(L, 2.0, 0.5)
    print(f"有效参数计算: 成功={success}, 消息='{msg}'")
    
    # 测试无效参数
    result, success, msg = safe_calc.safe_phoenix_calculation(L, -1.0, 0.5)
    print(f"无效参数计算: 成功={success}, 消息='{msg}'")
    
    # 单调性验证
    is_monotonic, msg = safe_calc.safe_phoenix_validation(2.0, 0.5)
    print(f"单调性验证: 通过={is_monotonic}, 消息='{msg}'")
    
    # 系统状态
    status = safe_calc.get_system_status()
    print(f"系统状态: 稳定={status['system_stable']}, 错误数={status['error_count']}")


def create_visualization():
    """创建可视化图表"""
    print("\n=== 创建可视化 ===")
    
    calculator = PhoenixCurveCalculator()
    
    # 不同参数的曲线对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 不同p值的影响
    L = np.linspace(0, 1, 256)
    p_values = [0.5, 1.0, 2.0, 4.0]
    
    ax1.set_title('不同p值的Phoenix曲线 (a=0.5)')
    ax1.plot(L, L, 'k--', alpha=0.5, label='恒等线')
    
    for p in p_values:
        L_out = calculator.compute_phoenix_curve(L, p, 0.5)
        ax1.plot(L, L_out, label=f'p={p}')
        
    ax1.set_xlabel('输入亮度 (PQ域)')
    ax1.set_ylabel('输出亮度 (PQ域)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 不同a值的影响
    a_values = [0.1, 0.3, 0.5, 0.8]
    
    ax2.set_title('不同a值的Phoenix曲线 (p=2.0)')
    ax2.plot(L, L, 'k--', alpha=0.5, label='恒等线')
    
    for a in a_values:
        L_out = calculator.compute_phoenix_curve(L, 2.0, a)
        ax2.plot(L, L_out, label=f'a={a}')
        
    ax2.set_xlabel('输入亮度 (PQ域)')
    ax2.set_ylabel('输出亮度 (PQ域)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phoenix_curves_demo.png', dpi=150, bbox_inches='tight')
    print("可视化图表已保存为 'phoenix_curves_demo.png'")


def main():
    """主函数"""
    print("HDR色调映射专利可视化工具 - 核心功能演示")
    print("=" * 50)
    
    try:
        # 演示各个模块
        demo_phoenix_curve()
        demo_pq_conversion()
        demo_safe_calculation()
        
        # 创建可视化 (需要matplotlib)
        try:
            import matplotlib.pyplot as plt
            create_visualization()
        except ImportError:
            print("\n注意: matplotlib未安装，跳过可视化演示")
            print("安装命令: pip install matplotlib")
            
        print("\n" + "=" * 50)
        print("核心功能演示完成！")
        print("所有模块工作正常，可以开始构建Gradio界面。")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()