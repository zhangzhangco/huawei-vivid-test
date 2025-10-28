#!/usr/bin/env python3
"""
数据导出和诊断功能演示
展示1D LUT导出、CSV导出和完整诊断包生成功能
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import (
    PhoenixCurveCalculator, QualityMetricsCalculator, 
    SessionState, TemporalStateData, ImageStats,
    ExportManager, CurveData, QualityMetrics, ExportMetadata,
    get_export_manager
)


def create_sample_data():
    """创建示例数据用于演示"""
    print("创建示例数据...")
    
    # 创建Phoenix曲线数据
    calculator = PhoenixCurveCalculator()
    L_input = np.linspace(0, 1, 512)
    p, a = 2.0, 0.5
    L_output = calculator.compute_phoenix_curve(L_input, p, a)
    
    # 创建曲线数据对象
    curve_data = CurveData(
        input_luminance=L_input,
        output_luminance=L_output,
        phoenix_curve=L_output,
        identity_line=L_input,
        curve_parameters={"p": p, "a": a}
    )
    
    # 创建会话状态
    session_state = SessionState(
        p=p,
        a=a,
        mode="艺术模式",
        luminance_channel="MaxRGB",
        dt_low=0.05,
        dt_high=0.10,
        enable_spline=False,
        th1=0.2,
        th2=0.5,
        th3=0.8,
        th_strength=0.0,
        window_size=9,
        lambda_smooth=0.3
    )
    
    # 创建时域状态
    temporal_state = TemporalStateData()
    # 添加一些示例历史数据
    for i in range(5):
        temporal_state.parameter_history.append((p + 0.1*i, a + 0.05*i))
        temporal_state.distortion_history.append(0.03 + 0.01*i)
        temporal_state.timestamp_history.append(1000.0 + i)
    temporal_state.total_frames = 5
    temporal_state.variance_reduction = 25.0
    temporal_state.smoothing_active = True
    
    # 创建质量指标
    quality_metrics = QualityMetrics(
        perceptual_distortion=0.035,
        local_contrast=0.12,
        variance_distortion=0.08,
        recommended_mode="自动模式",
        computation_time=0.15,
        is_monotonic=True,
        endpoint_error=1e-5,
        luminance_channel="MaxRGB"
    )
    
    # 创建图像统计信息
    image_stats = ImageStats(
        min_pq=0.01,
        max_pq=0.95,
        avg_pq=0.45,
        var_pq=0.08,
        input_format="OpenEXR(linear)",
        processing_path="Linear→PQ",
        pixel_count=1048576
    )
    
    return curve_data, session_state, temporal_state, quality_metrics, image_stats


def demo_lut_export():
    """演示1D LUT导出功能"""
    print("\n=== 1D LUT导出演示 ===")
    
    curve_data, session_state, _, _, _ = create_sample_data()
    export_manager = get_export_manager()
    
    # 导出LUT文件
    lut_filename = "demo_tone_mapping.cube"
    success = export_manager.export_lut(curve_data, session_state, lut_filename, samples=1024)
    
    if success:
        print(f"✓ 1D LUT已导出到: {lut_filename}")
        
        # 显示文件信息
        summary = export_manager.get_export_summary(lut_filename)
        print(f"  文件大小: {summary['file_size']} 字节")
        print(f"  估算采样点数: {summary.get('estimated_samples', 'N/A')}")
        print(f"  文件哈希: {summary['file_hash'][:16]}...")
        
        # 验证导出一致性
        is_consistent, max_error = export_manager.validate_export_consistency(
            curve_data.output_luminance, lut_filename, "lut"
        )
        print(f"  一致性验证: {'通过' if is_consistent else '失败'}")
        print(f"  最大误差: {max_error:.2e}")
        
        # 显示文件内容预览
        print("\n文件内容预览:")
        with open(lut_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:15]):  # 显示前15行
                print(f"  {line.rstrip()}")
            if len(lines) > 15:
                print(f"  ... (共{len(lines)}行)")
                
    else:
        print("✗ 1D LUT导出失败")


def demo_csv_export():
    """演示CSV导出功能"""
    print("\n=== CSV导出演示 ===")
    
    curve_data, session_state, _, quality_metrics, image_stats = create_sample_data()
    export_manager = get_export_manager()
    
    # 创建导出元数据
    metadata = ExportMetadata(
        export_time="2024-01-01T12:00:00",
        version="1.0",
        source_system="HDR色调映射专利可视化工具",
        parameters=session_state.to_dict(),
        image_stats=image_stats.__dict__ if hasattr(image_stats, '__dict__') else None,
        quality_metrics=quality_metrics.to_dict()
    )
    
    # 导出CSV文件
    csv_filename = "demo_curve_data.csv"
    success = export_manager.export_csv(curve_data, session_state, csv_filename, metadata)
    
    if success:
        print(f"✓ 曲线数据已导出到: {csv_filename}")
        
        # 显示文件信息
        summary = export_manager.get_export_summary(csv_filename)
        print(f"  文件大小: {summary['file_size']} 字节")
        print(f"  估算数据行数: {summary.get('estimated_rows', 'N/A')}")
        
        # 验证导出一致性
        is_consistent, max_error = export_manager.validate_export_consistency(
            curve_data.output_luminance, csv_filename, "csv"
        )
        print(f"  一致性验证: {'通过' if is_consistent else '失败'}")
        print(f"  最大误差: {max_error:.2e}")
        
        # 显示文件内容预览
        print("\n文件内容预览:")
        with open(csv_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20]):  # 显示前20行
                print(f"  {line.rstrip()}")
            if len(lines) > 20:
                print(f"  ... (共{len(lines)}行)")
                
    else:
        print("✗ CSV导出失败")


def demo_diagnostic_package():
    """演示诊断包生成功能"""
    print("\n=== 诊断包生成演示 ===")
    
    curve_data, session_state, temporal_state, quality_metrics, image_stats = create_sample_data()
    export_manager = get_export_manager()
    
    # 创建诊断包
    package_path = export_manager.create_diagnostic_package(
        curve_data, session_state, temporal_state, quality_metrics, image_stats
    )
    
    if package_path:
        print(f"✓ 诊断包已创建: {package_path}")
        
        # 显示包信息
        summary = export_manager.get_export_summary(package_path)
        print(f"  包大小: {summary['file_size']} 字节")
        print(f"  创建时间: {summary['created_time']}")
        
        # 显示包内容
        if 'archive_contents' in summary:
            print("\n包内容:")
            for item in summary['archive_contents']:
                print(f"  {item}")
                
        # 验证包完整性
        print(f"\n包完整性:")
        print(f"  文件哈希: {summary['file_hash'][:16]}...")
        
        # 检查必需文件
        required_files = [
            'README.md',
            'system_info.json',
            'config/session_config.json',
            'config/temporal_config.json',
            'analysis/quality_metrics.json',
            'curve_data.csv',
            'tone_mapping.cube'
        ]
        
        missing_files = []
        if 'archive_contents' in summary:
            for required in required_files:
                if required not in summary['archive_contents']:
                    missing_files.append(required)
                    
        if missing_files:
            print(f"  ⚠ 缺少文件: {missing_files}")
        else:
            print("  ✓ 所有必需文件都存在")
            
    else:
        print("✗ 诊断包创建失败")


def demo_export_validation():
    """演示导出验证功能"""
    print("\n=== 导出验证演示 ===")
    
    curve_data, session_state, _, _, _ = create_sample_data()
    export_manager = get_export_manager()
    
    # 创建测试文件
    test_lut = "test_validation.cube"
    test_csv = "test_validation.csv"
    
    # 导出文件
    export_manager.export_lut(curve_data, session_state, test_lut, samples=256)
    export_manager.export_csv(curve_data, session_state, test_csv)
    
    print("验证导出文件一致性:")
    
    # 验证LUT
    is_consistent, max_error = export_manager.validate_export_consistency(
        curve_data.output_luminance, test_lut, "lut"
    )
    print(f"  LUT一致性: {'通过' if is_consistent else '失败'} (误差: {max_error:.2e})")
    
    # 验证CSV
    is_consistent, max_error = export_manager.validate_export_consistency(
        curve_data.output_luminance, test_csv, "csv"
    )
    print(f"  CSV一致性: {'通过' if is_consistent else '失败'} (误差: {max_error:.2e})")
    
    # 清理测试文件
    for file in [test_lut, test_csv]:
        if os.path.exists(file):
            os.remove(file)
            print(f"  已清理测试文件: {file}")


def main():
    """主演示函数"""
    print("HDR色调映射数据导出和诊断功能演示")
    print("=" * 50)
    
    try:
        # 创建输出目录
        os.makedirs("exports", exist_ok=True)
        
        # 运行各项演示
        demo_lut_export()
        demo_csv_export()
        demo_diagnostic_package()
        demo_export_validation()
        
        print("\n" + "=" * 50)
        print("演示完成！")
        print("\n生成的文件:")
        for file in ["demo_tone_mapping.cube", "demo_curve_data.csv"]:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  {file} ({size} 字节)")
                
        # 检查exports目录
        exports_dir = Path("exports")
        if exports_dir.exists():
            export_files = list(exports_dir.glob("*.zip"))
            if export_files:
                print(f"\n诊断包:")
                for file in export_files:
                    print(f"  {file} ({file.stat().st_size} 字节)")
                    
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()