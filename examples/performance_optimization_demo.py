#!/usr/bin/env python3
"""
性能优化功能演示
展示GPU/Numba加速检测、自动降采样、进度指示和采样密度优化功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.core.performance_monitor import (
    get_performance_monitor, get_auto_downsampler, get_sampling_optimizer
)
from src.core.progress_handler import get_progress_handler, create_gradio_progress_callback
from src.core.phoenix_calculator import PhoenixCurveCalculator


def demo_acceleration_detection():
    """演示加速支持检测"""
    print("=" * 60)
    print("加速支持检测演示")
    print("=" * 60)
    
    monitor = get_performance_monitor()
    
    # 获取加速状态
    acceleration_status = monitor.get_acceleration_status()
    
    print(f"Numba 可用: {acceleration_status.numba_available}")
    if acceleration_status.numba_available:
        print(f"  版本: {acceleration_status.numba_version}")
        
    print(f"CUDA 可用: {acceleration_status.cuda_available}")
    if acceleration_status.cuda_available:
        print(f"  版本: {acceleration_status.cuda_version}")
        print(f"  GPU 数量: {acceleration_status.gpu_count}")
        print(f"  GPU 内存: {acceleration_status.gpu_memory_mb} MB")
        
    print(f"MKL 可用: {acceleration_status.mkl_available}")
    print(f"加速激活: {acceleration_status.acceleration_active}")
    
    if acceleration_status.fallback_reason:
        print(f"回退原因: {acceleration_status.fallback_reason}")
        
    print(f"\n加速状态摘要: {monitor.get_acceleration_summary()}")


def demo_performance_monitoring():
    """演示性能监控功能"""
    print("\n" + "=" * 60)
    print("性能监控演示")
    print("=" * 60)
    
    monitor = get_performance_monitor()
    phoenix_calc = PhoenixCurveCalculator()
    
    # 使用性能监控装饰器
    @monitor.measure_operation("phoenix_curve_calculation")
    def calculate_phoenix_curve(p: float, a: float, samples: int):
        """计算Phoenix曲线（带性能监控）"""
        L = np.linspace(0, 1, samples)
        return phoenix_calc.compute_phoenix_curve(L, p, a)
    
    print("执行多次Phoenix曲线计算...")
    
    # 执行不同复杂度的计算
    test_cases = [
        (2.0, 0.5, 256),   # 简单
        (2.2, 0.3, 512),   # 中等
        (2.5, 0.7, 1024),  # 复杂
        (3.0, 0.4, 2048),  # 很复杂
    ]
    
    for i, (p, a, samples) in enumerate(test_cases, 1):
        print(f"  测试 {i}: p={p}, a={a}, samples={samples}")
        result = calculate_phoenix_curve(p, a, samples)
        print(f"    结果长度: {len(result)}")
        
    # 获取性能摘要
    summary = monitor.get_performance_summary()
    print(f"\n性能摘要:")
    print(f"  总操作数: {summary['total_operations']}")
    print(f"  平均时间: {summary['average_duration_ms']:.2f} ms")
    print(f"  成功率: {summary['success_rate']:.1f}%")
    print(f"  峰值内存: {summary['memory_peak_mb']:.2f} MB")
    print(f"  平均CPU: {summary['cpu_average_percent']:.1f}%")
    
    # 检查性能警告
    warnings = monitor.check_performance_warnings()
    if warnings:
        print(f"\n性能警告:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print(f"\n✅ 无性能警告")


def demo_auto_downsampling():
    """演示自动降采样功能"""
    print("\n" + "=" * 60)
    print("自动降采样演示")
    print("=" * 60)
    
    downsampler = get_auto_downsampler()
    
    # 测试不同尺寸的图像
    test_images = [
        (800, 600, 3, "小图像"),
        (1920, 1080, 3, "Full HD"),
        (2560, 1440, 3, "2K"),
        (3840, 2160, 3, "4K"),
        (7680, 4320, 3, "8K"),
    ]
    
    print("测试不同尺寸图像的降采样决策:")
    
    for h, w, c, description in test_images:
        shape = (h, w, c)
        total_pixels = h * w
        
        should_downsample, scale, reason = downsampler.should_downsample(shape)
        
        print(f"\n{description} ({h}x{w}):")
        print(f"  像素总数: {total_pixels:,}")
        print(f"  需要降采样: {should_downsample}")
        print(f"  缩放比例: {scale:.3f}")
        print(f"  原因: {reason}")
        
        if should_downsample:
            new_h, new_w = int(h * scale), int(w * scale)
            new_pixels = new_h * new_w
            print(f"  新尺寸: {new_h}x{new_w}")
            print(f"  新像素数: {new_pixels:,}")
            print(f"  减少比例: {(1 - new_pixels/total_pixels)*100:.1f}%")
    
    # 模拟性能历史对降采样的影响
    print(f"\n模拟性能历史影响:")
    
    # 添加一些慢速处理历史
    for i in range(5):
        duration = 400 + i * 50  # 逐渐变慢
        pixels = 1920 * 1080
        downsampler.update_performance_history(duration, pixels)
        
    print(f"添加了5次慢速处理历史...")
    
    # 再次测试Full HD图像
    shape = (1920, 1080, 3)
    should_downsample, scale, reason = downsampler.should_downsample(shape)
    
    print(f"Full HD 图像 (基于性能历史):")
    print(f"  需要降采样: {should_downsample}")
    print(f"  缩放比例: {scale:.3f}")
    print(f"  原因: {reason}")
    
    # 获取降采样统计
    stats = downsampler.get_downsampling_stats()
    print(f"\n降采样统计:")
    print(f"  总操作数: {stats['total_operations']}")
    print(f"  平均时间: {stats['average_duration_ms']:.1f} ms")
    print(f"  平均像素数: {stats['average_pixels']:,}")
    print(f"  降采样率: {stats['downsampling_rate']:.1f}%")


def demo_sampling_density_optimization():
    """演示采样密度优化功能"""
    print("\n" + "=" * 60)
    print("采样密度优化演示")
    print("=" * 60)
    
    monitor = get_performance_monitor()
    optimizer = get_sampling_optimizer()
    phoenix_calc = PhoenixCurveCalculator()
    
    print("初始采样配置:")
    config = optimizer.get_current_sampling_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 模拟一些曲线计算性能数据
    print(f"\n模拟曲线计算性能数据...")
    
    @monitor.measure_operation("curve_calculation")
    def simulate_curve_calculation(samples: int):
        """模拟曲线计算"""
        # 模拟计算时间与采样点数成正比
        time.sleep(samples / 10000.0)  # 简单的时间模拟
        L = np.linspace(0, 1, samples)
        return phoenix_calc.compute_phoenix_curve(L, 2.0, 0.5)
    
    # 执行一些计算来建立性能历史
    test_samples = [256, 512, 1024, 1024, 2048]  # 逐渐增加复杂度
    
    for i, samples in enumerate(test_samples, 1):
        print(f"  执行计算 {i}: {samples} 采样点")
        result = simulate_curve_calculation(samples)
        
    # 获取优化后的采样密度
    print(f"\n优化后的采样密度:")
    
    optimized_display = optimizer.optimize_sampling_density("display")
    optimized_validation = optimizer.optimize_sampling_density("validation")
    
    print(f"  显示采样点数: {config['display_samples']} → {optimized_display}")
    print(f"  验证采样点数: {config['validation_samples']} → {optimized_validation}")
    
    # 显示性能摘要
    summary = monitor.get_performance_summary()
    print(f"\n曲线计算性能:")
    print(f"  平均时间: {summary['average_duration_ms']:.2f} ms")
    print(f"  目标时间: {optimizer.target_curve_time_ms} ms")
    print(f"  最大时间: {optimizer.max_curve_time_ms} ms")
    
    if summary['average_duration_ms'] > optimizer.target_curve_time_ms:
        print(f"  📉 性能较慢，建议减少采样点数")
    elif summary['average_duration_ms'] < optimizer.target_curve_time_ms:
        print(f"  📈 性能良好，可以增加采样点数")
    else:
        print(f"  ✅ 性能适中")


def demo_progress_handling():
    """演示进度处理功能"""
    print("\n" + "=" * 60)
    print("进度处理演示")
    print("=" * 60)
    
    progress_handler = get_progress_handler()
    
    # 创建进度回调函数
    def progress_callback(update):
        # 创建进度条显示
        progress_bar_length = 30
        filled_length = int(progress_bar_length * update.progress)
        bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
        
        print(f"\r[{bar}] {update.progress:.1%} - {update.description}", end='', flush=True)
        
        if update.progress >= 1.0:
            print()  # 完成时换行
    
    print("演示曲线计算进度:")
    
    # 演示曲线计算进度
    result = progress_handler.process_curve_with_progress(
        p=2.0,
        a=0.5,
        enable_spline=False,
        progress_callback=progress_callback
    )
    
    if result['success']:
        print(f"✅ 曲线计算成功")
        print(f"   处理时间: {result['processing_info']['processing_time_ms']:.1f} ms")
        print(f"   采样点数: {result['sampling_info']['display_samples']}")
        print(f"   单调性: {'通过' if result['is_monotonic'] else '失败'}")
    else:
        print(f"❌ 曲线计算失败: {result['error']}")
    
    print(f"\n演示图像处理进度:")
    
    # 创建测试图像
    test_image = np.random.rand(1000, 800, 3).astype(np.float32)
    
    def tone_curve_func(L):
        return np.clip(L ** 2.0 / (L ** 2.0 + 0.5 ** 2.0), 0, 1)
    
    # 演示图像处理进度
    result = progress_handler.process_image_with_progress(
        image=test_image,
        tone_curve_func=tone_curve_func,
        luminance_channel="MaxRGB",
        progress_callback=progress_callback
    )
    
    if result['success']:
        print(f"✅ 图像处理成功")
        info = result['processing_info']
        print(f"   原始尺寸: {info['original_shape']}")
        print(f"   最终尺寸: {info['final_shape']}")
        print(f"   处理时间: {info['processing_time_ms']:.1f} ms")
        print(f"   是否降采样: {info['downsampled']}")
        if info['downsampled']:
            print(f"   降采样原因: {info['downsample_reason']}")
            print(f"   缩放比例: {info['scale_factor']:.3f}")
    else:
        print(f"❌ 图像处理失败: {result['error']}")


def demo_integrated_workflow():
    """演示集成工作流"""
    print("\n" + "=" * 60)
    print("集成工作流演示")
    print("=" * 60)
    
    monitor = get_performance_monitor()
    
    print("执行完整的HDR处理工作流...")
    
    # 1. 检查系统加速能力
    print(f"\n1. 系统加速状态:")
    print(f"   {monitor.get_acceleration_summary()}")
    
    # 2. 创建测试图像
    print(f"\n2. 创建测试图像 (2000x1500)...")
    test_image = np.random.rand(2000, 1500, 3).astype(np.float32)
    
    # 3. 检查是否需要降采样
    downsampler = get_auto_downsampler()
    should_downsample, scale, reason = downsampler.should_downsample(test_image.shape)
    
    print(f"3. 降采样检查:")
    print(f"   需要降采样: {should_downsample}")
    print(f"   缩放比例: {scale:.3f}")
    print(f"   原因: {reason}")
    
    # 4. 优化采样密度
    optimizer = get_sampling_optimizer()
    display_samples = optimizer.optimize_sampling_density("display")
    validation_samples = optimizer.optimize_sampling_density("validation")
    
    print(f"\n4. 采样密度优化:")
    print(f"   显示采样: {display_samples}")
    print(f"   验证采样: {validation_samples}")
    
    # 5. 执行处理（模拟）
    print(f"\n5. 执行处理...")
    
    start_time = time.time()
    
    # 模拟图像处理
    if should_downsample:
        processed_image = downsampler.downsample_image(test_image, scale)
        print(f"   图像已降采样到: {processed_image.shape}")
    else:
        processed_image = test_image
        
    # 模拟曲线计算
    phoenix_calc = PhoenixCurveCalculator()
    L = np.linspace(0, 1, display_samples)
    curve = phoenix_calc.compute_phoenix_curve(L, 2.0, 0.5)
    
    processing_time = (time.time() - start_time) * 1000
    
    print(f"   处理完成，用时: {processing_time:.1f} ms")
    
    # 6. 更新性能历史
    total_pixels = test_image.shape[0] * test_image.shape[1]
    downsampler.update_performance_history(processing_time, total_pixels)
    
    # 7. 获取最终统计
    print(f"\n6. 最终统计:")
    
    perf_summary = monitor.get_performance_summary()
    print(f"   总操作数: {perf_summary['total_operations']}")
    print(f"   平均处理时间: {perf_summary['average_duration_ms']:.1f} ms")
    print(f"   成功率: {perf_summary['success_rate']:.1f}%")
    
    downsample_stats = downsampler.get_downsampling_stats()
    print(f"   降采样率: {downsample_stats['downsampling_rate']:.1f}%")
    
    warnings = monitor.check_performance_warnings()
    if warnings:
        print(f"\n⚠️  性能警告:")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print(f"\n✅ 无性能问题")


def main():
    """主函数"""
    print("HDR色调映射性能优化功能演示")
    print("=" * 60)
    
    try:
        # 1. 加速检测演示
        demo_acceleration_detection()
        
        # 2. 性能监控演示
        demo_performance_monitoring()
        
        # 3. 自动降采样演示
        demo_auto_downsampling()
        
        # 4. 采样密度优化演示
        demo_sampling_density_optimization()
        
        # 5. 进度处理演示
        demo_progress_handling()
        
        # 6. 集成工作流演示
        demo_integrated_workflow()
        
        print(f"\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print(f"\n\n演示被用户中断")
    except Exception as e:
        print(f"\n\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()