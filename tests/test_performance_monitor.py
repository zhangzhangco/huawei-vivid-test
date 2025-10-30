"""
性能监控模块测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from src.core.performance_monitor import (
    PerformanceMonitor, AccelerationDetector, AutoDownsampler, 
    SamplingDensityOptimizer, PerformanceMetrics, AccelerationStatus
)


class TestAccelerationDetector:
    """加速检测器测试"""
    
    def test_detect_acceleration_support(self):
        """测试加速支持检测"""
        detector = AccelerationDetector()
        
        # 重置检测状态
        detector.reset_detection()
        
        status = detector.detect_acceleration_support()
        
        assert isinstance(status, AccelerationStatus)
        assert isinstance(status.numba_available, bool)
        assert isinstance(status.cuda_available, bool)
        assert isinstance(status.mkl_available, bool)
        assert isinstance(status.acceleration_active, bool)
        
    def test_get_acceleration_summary(self):
        """测试加速状态摘要"""
        detector = AccelerationDetector()
        summary = detector.get_acceleration_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
    @patch('importlib.util.find_spec')
    def test_numba_not_available(self, mock_find_spec):
        """测试Numba不可用的情况"""
        mock_find_spec.return_value = None
        
        detector = AccelerationDetector()
        detector.reset_detection()
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'numba'")):
            status = detector.detect_acceleration_support()
            
        assert not status.numba_available
        assert not status.cuda_available
        assert "Numba not installed" in status.fallback_reason


class TestPerformanceMonitor:
    """性能监控器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        monitor = PerformanceMonitor(max_history=50)
        
        assert monitor.max_history == 50
        assert len(monitor.metrics_history) == 0
        assert isinstance(monitor.thresholds, dict)
        
    def test_measure_operation_decorator(self):
        """测试操作测量装饰器"""
        monitor = PerformanceMonitor()
        
        @monitor.measure_operation("test_operation")
        def test_function(x, y):
            time.sleep(0.01)  # 模拟一些处理时间
            return x + y
            
        result = test_function(1, 2)
        
        assert result == 3
        assert len(monitor.metrics_history) == 1
        
        metrics = monitor.metrics_history[0]
        assert metrics.operation_name == "test_operation"
        assert metrics.success
        assert metrics.duration_ms > 0
        
    def test_measure_operation_with_exception(self):
        """测试带异常的操作测量"""
        monitor = PerformanceMonitor()
        
        @monitor.measure_operation("failing_operation")
        def failing_function():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            failing_function()
            
        assert len(monitor.metrics_history) == 1
        
        metrics = monitor.metrics_history[0]
        assert not metrics.success
        assert metrics.error_message == "Test error"
        
    def test_get_performance_summary(self):
        """测试性能摘要"""
        monitor = PerformanceMonitor()
        
        # 添加一些测试指标
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_name=f"test_op_{i}",
                start_time=time.time(),
                end_time=time.time() + 0.1,
                duration_ms=100 + i * 10,
                memory_usage_mb=10 + i,
                cpu_usage_percent=50 + i * 5,
                success=True
            )
            monitor._add_metrics(metrics)
            
        summary = monitor.get_performance_summary()
        
        assert summary['total_operations'] == 5
        assert summary['average_duration_ms'] > 0
        assert summary['success_rate'] == 100
        assert len(summary['recent_operations']) == 5
        
    def test_check_performance_warnings(self):
        """测试性能警告检查"""
        monitor = PerformanceMonitor()
        
        # 添加一个响应时间过长的曲线操作
        slow_metrics = PerformanceMetrics(
            operation_name="curve_update",
            start_time=time.time(),
            end_time=time.time() + 1,
            duration_ms=1000,  # 超过500ms阈值
            memory_usage_mb=10,
            cpu_usage_percent=50,
            success=True
        )
        monitor._add_metrics(slow_metrics)
        
        warnings = monitor.check_performance_warnings()
        
        assert len(warnings) > 0
        assert any("曲线更新响应时间过长" in warning for warning in warnings)


class TestAutoDownsampler:
    """自动降采样器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        monitor = PerformanceMonitor()
        downsampler = AutoDownsampler(monitor)
        
        assert downsampler.max_pixels == 1280 * 1280
        assert downsampler.target_response_time_ms == 300
        assert downsampler.min_scale == 0.25
        
    def test_should_downsample_large_image(self):
        """测试大图像降采样判断"""
        monitor = PerformanceMonitor()
        downsampler = AutoDownsampler(monitor)
        
        # 测试超大图像
        large_shape = (2000, 2000, 3)
        should_downsample, scale, reason = downsampler.should_downsample(large_shape)
        
        assert should_downsample
        assert 0 < scale < 1
        assert "图像过大" in reason
        
    def test_should_downsample_small_image(self):
        """测试小图像不需要降采样"""
        monitor = PerformanceMonitor()
        downsampler = AutoDownsampler(monitor)
        
        # 测试小图像
        small_shape = (800, 600, 3)
        should_downsample, scale, reason = downsampler.should_downsample(small_shape)
        
        assert not should_downsample
        assert scale == 1.0
        assert "无需降采样" in reason
        
    def test_downsample_image(self):
        """测试图像降采样"""
        monitor = PerformanceMonitor()
        downsampler = AutoDownsampler(monitor)
        
        # 创建测试图像
        original_image = np.random.rand(1000, 800, 3).astype(np.float32)
        scale = 0.5
        
        downsampled = downsampler.downsample_image(original_image, scale)
        
        expected_h, expected_w = int(1000 * scale), int(800 * scale)
        assert downsampled.shape[:2] == (expected_h, expected_w)
        assert downsampled.shape[2] == 3
        
    def test_update_performance_history(self):
        """测试性能历史更新"""
        monitor = PerformanceMonitor()
        downsampler = AutoDownsampler(monitor)
        
        # 添加性能数据
        downsampler.update_performance_history(250, 1000000)
        downsampler.update_performance_history(350, 2000000)
        
        assert len(downsampler.recent_performance) == 2
        
        stats = downsampler.get_downsampling_stats()
        assert stats['total_operations'] == 2
        assert stats['average_duration_ms'] == 300
        

class TestSamplingDensityOptimizer:
    """采样密度优化器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        monitor = PerformanceMonitor()
        optimizer = SamplingDensityOptimizer(monitor)
        
        assert optimizer.base_display_samples == 512
        assert optimizer.base_validation_samples == 1024
        assert optimizer.current_display_samples == 512
        assert optimizer.current_validation_samples == 1024
        
    def test_optimize_sampling_density_no_history(self):
        """测试无历史数据时的采样密度优化"""
        monitor = PerformanceMonitor()
        optimizer = SamplingDensityOptimizer(monitor)
        
        # 无历史数据时应返回默认值
        display_samples = optimizer.optimize_sampling_density("display")
        validation_samples = optimizer.optimize_sampling_density("validation")
        
        assert display_samples == optimizer.base_display_samples
        assert validation_samples == optimizer.base_validation_samples
        
    def test_optimize_sampling_density_with_slow_performance(self):
        """测试性能较慢时的采样密度优化"""
        monitor = PerformanceMonitor()
        optimizer = SamplingDensityOptimizer(monitor)
        
        # 添加一些慢速的曲线计算历史
        for _ in range(3):
            slow_metrics = PerformanceMetrics(
                operation_name="curve_calculation",
                start_time=time.time(),
                end_time=time.time() + 0.6,
                duration_ms=600,  # 超过最大阈值
                memory_usage_mb=10,
                cpu_usage_percent=50,
                success=True
            )
            monitor._add_metrics(slow_metrics)
            
        # 优化后应该减少采样点数
        new_display_samples = optimizer.optimize_sampling_density("display")
        
        assert new_display_samples < optimizer.base_display_samples
        assert new_display_samples >= optimizer.min_samples
        
    def test_get_current_sampling_config(self):
        """测试获取当前采样配置"""
        monitor = PerformanceMonitor()
        optimizer = SamplingDensityOptimizer(monitor)
        
        config = optimizer.get_current_sampling_config()
        
        assert 'display_samples' in config
        assert 'validation_samples' in config
        assert 'base_display_samples' in config
        assert 'base_validation_samples' in config
        assert 'min_samples' in config
        assert 'max_samples' in config
        
    def test_reset_sampling_density(self):
        """测试重置采样密度"""
        monitor = PerformanceMonitor()
        optimizer = SamplingDensityOptimizer(monitor)
        
        # 修改当前采样密度
        optimizer.current_display_samples = 256
        optimizer.current_validation_samples = 512
        
        # 重置
        optimizer.reset_sampling_density()
        
        assert optimizer.current_display_samples == optimizer.base_display_samples
        assert optimizer.current_validation_samples == optimizer.base_validation_samples


class TestIntegration:
    """集成测试"""
    
    def test_global_instances(self):
        """测试全局实例获取"""
        from src.core.performance_monitor import (
            get_performance_monitor, get_auto_downsampler, get_sampling_optimizer
        )
        
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # 应该返回同一个实例
        assert monitor1 is monitor2
        
        downsampler1 = get_auto_downsampler()
        downsampler2 = get_auto_downsampler()
        
        assert downsampler1 is downsampler2
        
        optimizer1 = get_sampling_optimizer()
        optimizer2 = get_sampling_optimizer()
        
        assert optimizer1 is optimizer2
        
    def test_performance_monitoring_workflow(self):
        """测试完整的性能监控工作流"""
        from src.core.performance_monitor import get_performance_monitor, get_auto_downsampler
        
        monitor = get_performance_monitor()
        downsampler = get_auto_downsampler()
        
        # 模拟图像处理工作流
        test_image_shape = (1500, 1200, 3)
        
        # 检查是否需要降采样
        should_downsample, scale, reason = downsampler.should_downsample(test_image_shape)
        
        if should_downsample:
            # 模拟降采样处理时间
            processing_time = 200  # ms
            total_pixels = int(test_image_shape[0] * test_image_shape[1] * scale * scale)
        else:
            processing_time = 400  # ms
            total_pixels = test_image_shape[0] * test_image_shape[1]
            
        # 更新性能历史
        downsampler.update_performance_history(processing_time, total_pixels)
        
        # 获取统计信息
        stats = downsampler.get_downsampling_stats()
        
        assert stats['total_operations'] == 1
        assert stats['average_duration_ms'] == processing_time
        
        # 检查性能警告
        warnings = monitor.check_performance_warnings()
        
        # 根据处理时间可能会有警告
        if processing_time > 300:
            assert len(warnings) > 0