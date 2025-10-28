"""
性能监控模块
实现性能监控、GPU/Numba加速检测和自动降采样功能
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from threading import Lock
import sys
import importlib.util


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccelerationStatus:
    """加速状态数据类"""
    numba_available: bool = False
    numba_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    mkl_available: bool = False
    acceleration_active: bool = False
    fallback_reason: Optional[str] = None


class AccelerationDetector:
    """加速支持检测器"""
    
    def __init__(self):
        self._status = None
        self._detection_lock = Lock()
        
    def detect_acceleration_support(self) -> AccelerationStatus:
        """检测可用的加速支持"""
        
        with self._detection_lock:
            if self._status is not None:
                return self._status
                
            status = AccelerationStatus()
            
            # 检测Numba支持
            try:
                import numba
                status.numba_available = True
                status.numba_version = numba.__version__
                
                # 检测CUDA支持
                try:
                    from numba import cuda
                    if cuda.is_available():
                        status.cuda_available = True
                        status.gpu_count = len(cuda.gpus)
                        
                        # 获取GPU内存信息
                        try:
                            gpu = cuda.get_current_device()
                            status.gpu_memory_mb = gpu.memory_info.total // (1024 * 1024)
                        except Exception:
                            status.gpu_memory_mb = 0
                            
                        # 获取CUDA版本
                        try:
                            status.cuda_version = str(cuda.runtime.get_version())
                        except Exception:
                            status.cuda_version = "Unknown"
                            
                except ImportError:
                    status.cuda_available = False
                    status.fallback_reason = "CUDA not available in Numba"
                    
            except ImportError:
                status.numba_available = False
                status.fallback_reason = "Numba not installed"
                
            # 检测NumPy MKL支持
            try:
                import numpy as np
                if hasattr(np, '__config__'):
                    config = np.__config__.show()
                    if 'mkl' in str(config).lower():
                        status.mkl_available = True
            except Exception:
                status.mkl_available = False
                
            # 确定是否启用加速
            status.acceleration_active = (
                status.numba_available or 
                status.cuda_available or 
                status.mkl_available
            )
            
            self._status = status
            return status
            
    def get_acceleration_summary(self) -> str:
        """获取加速状态摘要"""
        status = self.detect_acceleration_support()
        
        summary_parts = []
        
        if status.numba_available:
            summary_parts.append(f"✓ Numba {status.numba_version}")
        else:
            summary_parts.append("✗ Numba")
            
        if status.cuda_available:
            summary_parts.append(f"✓ CUDA {status.cuda_version} ({status.gpu_count} GPU)")
        else:
            summary_parts.append("✗ CUDA")
            
        if status.mkl_available:
            summary_parts.append("✓ MKL")
        else:
            summary_parts.append("✗ MKL")
            
        summary = " | ".join(summary_parts)
        
        if status.fallback_reason:
            summary += f" | 回退原因: {status.fallback_reason}"
            
        return summary
        
    def reset_detection(self):
        """重置检测状态（用于测试）"""
        with self._detection_lock:
            self._status = None


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self._lock = Lock()
        self.acceleration_detector = AccelerationDetector()
        
        # 性能阈值配置
        self.thresholds = {
            'curve_update_ms': 500,
            'image_processing_ms': 300,
            'memory_warning_mb': 1000,
            'cpu_warning_percent': 80
        }
        
    def start_operation(self, operation_name: str) -> str:
        """开始监控操作"""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        return operation_id
        
    def measure_operation(self, operation_name: str):
        """操作性能测量装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                start_cpu = self._get_cpu_usage()
                
                success = True
                error_message = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    end_cpu = self._get_cpu_usage()
                    
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=(end_time - start_time) * 1000,
                        memory_usage_mb=max(end_memory - start_memory, 0),
                        cpu_usage_percent=(start_cpu + end_cpu) / 2,
                        success=success,
                        error_message=error_message
                    )
                    
                    self._add_metrics(metrics)
                    
                return result
            return wrapper
        return decorator
        
    def _add_metrics(self, metrics: PerformanceMetrics):
        """添加性能指标"""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # 保持历史记录在限制内
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
                
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
            
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self._lock:
            if not self.metrics_history:
                return {
                    'total_operations': 0,
                    'average_duration_ms': 0,
                    'success_rate': 100,
                    'memory_peak_mb': 0,
                    'cpu_average_percent': 0,
                    'recent_operations': []
                }
                
            recent_metrics = self.metrics_history[-10:]  # 最近10个操作
            
            durations = [m.duration_ms for m in recent_metrics]
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
            success_count = sum(1 for m in recent_metrics if m.success)
            
            return {
                'total_operations': len(self.metrics_history),
                'average_duration_ms': np.mean(durations) if durations else 0,
                'success_rate': (success_count / len(recent_metrics)) * 100 if recent_metrics else 100,
                'memory_peak_mb': max(memory_usage) if memory_usage else 0,
                'cpu_average_percent': np.mean(cpu_usage) if cpu_usage else 0,
                'recent_operations': [
                    {
                        'name': m.operation_name,
                        'duration_ms': m.duration_ms,
                        'success': m.success,
                        'timestamp': m.end_time
                    }
                    for m in recent_metrics
                ]
            }
            
    def check_performance_warnings(self) -> List[str]:
        """检查性能警告"""
        warnings = []
        
        with self._lock:
            if not self.metrics_history:
                return warnings
                
            recent_metrics = self.metrics_history[-5:]  # 最近5个操作
            
            # 检查响应时间
            for metrics in recent_metrics:
                if 'curve' in metrics.operation_name.lower():
                    if metrics.duration_ms > self.thresholds['curve_update_ms']:
                        warnings.append(f"曲线更新响应时间过长: {metrics.duration_ms:.1f}ms > {self.thresholds['curve_update_ms']}ms")
                        
                elif 'image' in metrics.operation_name.lower():
                    if metrics.duration_ms > self.thresholds['image_processing_ms']:
                        warnings.append(f"图像处理响应时间过长: {metrics.duration_ms:.1f}ms > {self.thresholds['image_processing_ms']}ms")
                        
            # 检查内存使用
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            if memory_usage and max(memory_usage) > self.thresholds['memory_warning_mb']:
                warnings.append(f"内存使用过高: {max(memory_usage):.1f}MB > {self.thresholds['memory_warning_mb']}MB")
                
            # 检查CPU使用
            cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
            if cpu_usage and np.mean(cpu_usage) > self.thresholds['cpu_warning_percent']:
                warnings.append(f"CPU使用率过高: {np.mean(cpu_usage):.1f}% > {self.thresholds['cpu_warning_percent']}%")
                
        return warnings
        
    def get_acceleration_status(self) -> AccelerationStatus:
        """获取加速状态"""
        return self.acceleration_detector.detect_acceleration_support()
        
    def get_acceleration_summary(self) -> str:
        """获取加速状态摘要"""
        return self.acceleration_detector.get_acceleration_summary()
        
    def reset_metrics(self):
        """重置性能指标"""
        with self._lock:
            self.metrics_history.clear()


class AutoDownsampler:
    """自动降采样器"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        
        # 降采样配置
        self.max_pixels = 1280 * 1280  # 1.6MP
        self.target_response_time_ms = 300
        self.min_scale = 0.25  # 最小缩放比例
        
        # 自适应参数
        self.recent_performance = []
        self.max_performance_history = 10
        
    def should_downsample(self, image_shape: Tuple[int, ...]) -> Tuple[bool, float, str]:
        """判断是否需要降采样
        
        Args:
            image_shape: 图像形状 (H, W, C) 或 (H, W)
            
        Returns:
            (需要降采样, 建议缩放比例, 原因)
        """
        if len(image_shape) < 2:
            return False, 1.0, "无效图像形状"
            
        h, w = image_shape[:2]
        total_pixels = h * w
        
        # 基于像素数的硬限制
        if total_pixels > self.max_pixels:
            scale = np.sqrt(self.max_pixels / total_pixels)
            scale = max(scale, self.min_scale)
            return True, scale, f"图像过大 ({total_pixels:,} > {self.max_pixels:,} 像素)"
            
        # 基于历史性能的自适应降采样
        if self.recent_performance:
            avg_duration = np.mean([p['duration_ms'] for p in self.recent_performance])
            if avg_duration > self.target_response_time_ms:
                # 根据性能历史调整缩放比例
                performance_ratio = self.target_response_time_ms / avg_duration
                scale = np.sqrt(performance_ratio)
                scale = max(scale, self.min_scale)
                
                if scale < 0.9:  # 只有在需要显著缩放时才降采样
                    return True, scale, f"性能优化 (平均响应时间 {avg_duration:.1f}ms > {self.target_response_time_ms}ms)"
                    
        return False, 1.0, "无需降采样"
        
    def downsample_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """降采样图像
        
        Args:
            image: 输入图像
            scale: 缩放比例
            
        Returns:
            降采样后的图像
        """
        if scale >= 1.0:
            return image
            
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 确保新尺寸至少为1
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        import cv2
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
    def update_performance_history(self, duration_ms: float, pixels: int):
        """更新性能历史"""
        self.recent_performance.append({
            'duration_ms': duration_ms,
            'pixels': pixels,
            'timestamp': time.time()
        })
        
        # 保持历史记录在限制内
        if len(self.recent_performance) > self.max_performance_history:
            self.recent_performance.pop(0)
            
    def get_downsampling_stats(self) -> Dict[str, Any]:
        """获取降采样统计"""
        if not self.recent_performance:
            return {
                'total_operations': 0,
                'average_duration_ms': 0,
                'average_pixels': 0,
                'downsampling_rate': 0
            }
            
        durations = [p['duration_ms'] for p in self.recent_performance]
        pixels = [p['pixels'] for p in self.recent_performance]
        
        # 计算降采样率（基于像素数变化）
        downsampling_count = sum(1 for p in pixels if p < self.max_pixels)
        downsampling_rate = (downsampling_count / len(pixels)) * 100
        
        return {
            'total_operations': len(self.recent_performance),
            'average_duration_ms': np.mean(durations),
            'average_pixels': int(np.mean(pixels)),
            'downsampling_rate': downsampling_rate,
            'max_pixels_limit': self.max_pixels,
            'target_response_time_ms': self.target_response_time_ms
        }


class SamplingDensityOptimizer:
    """采样密度优化器"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        
        # 采样密度配置
        self.base_display_samples = 512
        self.base_validation_samples = 1024
        self.min_samples = 256
        self.max_samples = 2048
        
        # 性能阈值
        self.target_curve_time_ms = 100  # 曲线计算目标时间
        self.max_curve_time_ms = 500     # 曲线计算最大时间
        
        # 自适应参数
        self.current_display_samples = self.base_display_samples
        self.current_validation_samples = self.base_validation_samples
        
    def optimize_sampling_density(self, operation_type: str = "display") -> int:
        """优化采样密度
        
        Args:
            operation_type: 操作类型 ("display" 或 "validation")
            
        Returns:
            优化后的采样点数
        """
        # 获取最近的曲线计算性能
        recent_metrics = []
        with self.performance_monitor._lock:
            for metrics in self.performance_monitor.metrics_history[-5:]:
                if 'curve' in metrics.operation_name.lower():
                    recent_metrics.append(metrics)
                    
        if not recent_metrics:
            # 没有历史数据，使用默认值
            return (self.base_display_samples if operation_type == "display" 
                   else self.base_validation_samples)
            
        # 计算平均响应时间
        avg_duration = np.mean([m.duration_ms for m in recent_metrics])
        
        # 根据性能调整采样密度
        if operation_type == "display":
            current_samples = self.current_display_samples
            
            if avg_duration > self.max_curve_time_ms:
                # 性能不佳，减少采样点
                new_samples = int(current_samples * 0.8)
            elif avg_duration < self.target_curve_time_ms:
                # 性能良好，可以增加采样点
                new_samples = int(current_samples * 1.2)
            else:
                new_samples = current_samples
                
            # 限制在合理范围内
            new_samples = max(self.min_samples, min(new_samples, self.max_samples))
            self.current_display_samples = new_samples
            
            return new_samples
            
        else:  # validation
            current_samples = self.current_validation_samples
            
            if avg_duration > self.max_curve_time_ms:
                new_samples = int(current_samples * 0.9)
            elif avg_duration < self.target_curve_time_ms:
                new_samples = int(current_samples * 1.1)
            else:
                new_samples = current_samples
                
            new_samples = max(self.min_samples, min(new_samples, self.max_samples * 2))
            self.current_validation_samples = new_samples
            
            return new_samples
            
    def get_current_sampling_config(self) -> Dict[str, int]:
        """获取当前采样配置"""
        return {
            'display_samples': self.current_display_samples,
            'validation_samples': self.current_validation_samples,
            'base_display_samples': self.base_display_samples,
            'base_validation_samples': self.base_validation_samples,
            'min_samples': self.min_samples,
            'max_samples': self.max_samples
        }
        
    def reset_sampling_density(self):
        """重置采样密度到默认值"""
        self.current_display_samples = self.base_display_samples
        self.current_validation_samples = self.base_validation_samples


# 全局性能监控器实例
_global_performance_monitor = None
_global_auto_downsampler = None
_global_sampling_optimizer = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def get_auto_downsampler() -> AutoDownsampler:
    """获取全局自动降采样器实例"""
    global _global_auto_downsampler, _global_performance_monitor
    if _global_auto_downsampler is None:
        if _global_performance_monitor is None:
            _global_performance_monitor = PerformanceMonitor()
        _global_auto_downsampler = AutoDownsampler(_global_performance_monitor)
    return _global_auto_downsampler


def get_sampling_optimizer() -> SamplingDensityOptimizer:
    """获取全局采样密度优化器实例"""
    global _global_sampling_optimizer, _global_performance_monitor
    if _global_sampling_optimizer is None:
        if _global_performance_monitor is None:
            _global_performance_monitor = PerformanceMonitor()
        _global_sampling_optimizer = SamplingDensityOptimizer(_global_performance_monitor)
    return _global_sampling_optimizer