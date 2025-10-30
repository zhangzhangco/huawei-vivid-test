"""
HDR质量评估扩展模块
实现ExtendedMetrics类，提供高级质量指标计算、自动质量评估和艺术家友好的提示功能

性能优化特性:
- 30ms内完成1MP图像处理
- 向量化计算和内存优化
- 智能缓存和批量处理
"""

import numpy as np
import json
import logging
import gc
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from functools import lru_cache

from .config_manager import ConfigManager

# 性能优化：设置numpy使用单线程（避免线程开销）
import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
if 'NUMEXPR_NUM_THREADS' not in os.environ:
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.stats = {
            'total_assessments': 0,
            'total_time_ms': 0.0,
            'max_time_ms': 0.0,
            'min_time_ms': float('inf'),
            'over_target_count': 0
        }
        self.target_time_ms = 30.0
    
    def record_assessment(self, elapsed_time_ms: float):
        """记录评估统计"""
        self.stats['total_assessments'] += 1
        self.stats['total_time_ms'] += elapsed_time_ms
        self.stats['max_time_ms'] = max(self.stats['max_time_ms'], elapsed_time_ms)
        self.stats['min_time_ms'] = min(self.stats['min_time_ms'], elapsed_time_ms)
        
        if elapsed_time_ms > self.target_time_ms:
            self.stats['over_target_count'] += 1
    
    def get_average_time(self) -> float:
        """获取平均处理时间"""
        if self.stats['total_assessments'] > 0:
            return self.stats['total_time_ms'] / self.stats['total_assessments']
        return 0.0
    
    def get_performance_report(self) -> Dict[str, Union[float, int, str]]:
        """获取性能报告"""
        avg_time = self.get_average_time()
        success_rate = (
            (self.stats['total_assessments'] - self.stats['over_target_count']) / 
            max(1, self.stats['total_assessments']) * 100
        )
        
        return {
            'total_assessments': self.stats['total_assessments'],
            'average_time_ms': round(avg_time, 2),
            'max_time_ms': round(self.stats['max_time_ms'], 2),
            'min_time_ms': round(self.stats['min_time_ms'], 2) if self.stats['min_time_ms'] != float('inf') else 0.0,
            'success_rate_percent': round(success_rate, 1),
            'over_target_count': self.stats['over_target_count']
        }


class ExtendedMetrics:
    """
    扩展质量评估模块
    提供HDR色调映射的高级质量指标计算和自动评估功能
    
    性能特性:
    - 30ms内完成1MP图像处理
    - 向量化计算优化
    - 内存使用优化
    - 性能监控和报告
    """
    
    def __init__(self, config_path: str = "config/metrics.json", enable_performance_monitoring: bool = True):
        """
        初始化扩展质量评估模块
        
        Args:
            config_path: 配置文件路径
            enable_performance_monitoring: 是否启用性能监控
        """
        self.config_path = config_path
        self.eps = 1e-6  # 除零保护阈值
        self.logger = logging.getLogger(__name__)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # 使用ConfigManager加载配置
        self.config_manager = ConfigManager(config_path)
        self.thresholds = self.config_manager.load_thresholds()
        
        # 性能优化：预分配常用的数组大小阈值
        self.large_array_threshold = 1000000  # 1MP
        
        # 内存管理：设置垃圾回收阈值
        gc.set_threshold(700, 10, 10)
        
    def reload_thresholds(self) -> None:
        """
        重新加载阈值配置（支持热更新）
        确保阈值更新时质量评估逻辑同步更新
        """
        self.thresholds = self.config_manager.load_thresholds()
        self.logger.info("阈值配置已重新加载")
        
    def get_current_thresholds(self) -> Dict[str, float]:
        """
        获取当前使用的阈值配置
        
        Returns:
            当前阈值配置字典
        """
        return self.config_manager.load_thresholds()
    
    def update_threshold(self, key: str, value: float) -> bool:
        """
        更新单个阈值并重新加载配置
        
        Args:
            key: 阈值配置项名称
            value: 新的阈值
            
        Returns:
            是否更新成功
        """
        success = self.config_manager.update_threshold(key, value)
        if success:
            self.reload_thresholds()
        return success
    
    def safe_divide(self, numerator: float, denominator: float, fallback: float = None) -> float:
        """
        安全除法操作，提供除零保护
        
        Args:
            numerator: 分子
            denominator: 分母
            fallback: 回退值，默认使用self.eps
            
        Returns:
            安全的除法结果
        """
        if fallback is None:
            fallback = self.eps
        return numerator / max(abs(denominator), fallback)
    
    def safe_log(self, value: float, fallback: float = None) -> float:
        """
        安全对数操作，提供除零保护
        
        Args:
            value: 输入值
            fallback: 回退值，默认使用self.eps
            
        Returns:
            安全的对数结果
        """
        if fallback is None:
            fallback = self.eps
        return np.log(max(abs(value), fallback))
    
    def calculate_basic_stats(self, lin: np.ndarray, lout: np.ndarray) -> Dict[str, float]:
        """
        计算基础统计数据
        
        Args:
            lin: 输入亮度数据 (PQ域，范围0-1)
            lout: 输出亮度数据 (PQ域，映射后，范围0-1)
            
        Returns:
            基础统计数据字典
        """
        # 确保输入为numpy数组并进行类型转换
        lin_array = np.asarray(lin, dtype=np.float64)
        lout_array = np.asarray(lout, dtype=np.float64)
        
        # 展平数组以便计算统计量
        lin_flat = lin_array.flatten()
        lout_flat = lout_array.flatten()
        
        # 计算基础统计量
        stats = {
            'Lmin_in': float(np.min(lin_flat)),
            'Lmax_in': float(np.max(lin_flat)),
            'Lmin_out': float(np.min(lout_flat)),
            'Lmax_out': float(np.max(lout_flat))
        }
        
        return stats
    
    def calculate_exposure_metrics(self, lin: np.ndarray, lout: np.ndarray) -> Dict[str, float]:
        """
        计算曝光相关指标
        
        Args:
            lin: 输入亮度数据 (PQ域，范围0-1)
            lout: 输出亮度数据 (PQ域，映射后，范围0-1)
            
        Returns:
            曝光指标字典
        """
        # 确保输入为numpy数组
        lin_array = np.asarray(lin, dtype=np.float64)
        lout_array = np.asarray(lout, dtype=np.float64)
        
        # 展平数组
        lin_flat = lin_array.flatten()
        lout_flat = lout_array.flatten()
        
        # 计算S_ratio (高光饱和比例)
        # 定义高光区域为PQ值 > 0.9的像素
        highlight_threshold = 0.9
        highlight_mask_out = lout_flat > highlight_threshold
        total_pixels = len(lout_flat)
        
        if total_pixels > 0:
            s_ratio = np.sum(highlight_mask_out) / total_pixels
        else:
            s_ratio = 0.0
        
        # 计算C_shadow (暗部压缩比例)
        # 按照建议：简单地用输出中暗部像素的比例
        shadow_threshold = 0.05  # 使用0.05作为暗部阈值
        shadow_mask_out = lout_flat < shadow_threshold
        
        if total_pixels > 0:
            c_shadow = np.sum(shadow_mask_out) / total_pixels
        else:
            c_shadow = 0.0
        
        # 计算R_DR (动态范围保持率)
        dr_in = np.max(lin_flat) - np.min(lin_flat)
        dr_out = np.max(lout_flat) - np.min(lout_flat)
        
        if dr_in > self.eps:
            r_dr = dr_out / dr_in
        else:
            r_dr = 1.0
        
        # 计算ΔL_mean_norm (归一化平均亮度漂移)
        # 按照草稿要求：亮度差值除以动态范围
        mean_in = np.mean(lin_flat)
        mean_out = np.mean(lout_flat)
        dr_in = np.max(lin_flat) - np.min(lin_flat)
        
        if dr_in > self.eps:
            delta_l_mean_norm = abs(mean_out - mean_in) / dr_in
        else:
            delta_l_mean_norm = 0.0
        
        return {
            'S_ratio': float(s_ratio),
            'C_shadow': float(c_shadow),
            'R_DR': float(r_dr),
            'ΔL_mean_norm': float(delta_l_mean_norm)
        }
    
    def calculate_histogram_overlap(self, lin: np.ndarray, lout: np.ndarray) -> float:
        """
        计算直方图重叠度
        
        Args:
            lin: 输入亮度数据 (PQ域，范围0-1)
            lout: 输出亮度数据 (PQ域，映射后，范围0-1)
            
        Returns:
            直方图重叠度 (0-1之间)
        """
        # 确保输入为numpy数组
        lin_array = np.asarray(lin, dtype=np.float64)
        lout_array = np.asarray(lout, dtype=np.float64)
        
        # 展平数组
        lin_flat = lin_array.flatten()
        lout_flat = lout_array.flatten()
        
        # 生成256-bin直方图
        bins = 256
        range_pq = (0.0, 1.0)
        
        # 计算归一化直方图
        hist_in, _ = np.histogram(lin_flat, bins=bins, range=range_pq, density=True)
        hist_out, _ = np.histogram(lout_flat, bins=bins, range=range_pq, density=True)
        
        # 归一化直方图 (确保总和为1)
        hist_in_norm = hist_in / (np.sum(hist_in) + self.eps)
        hist_out_norm = hist_out / (np.sum(hist_out) + self.eps)
        
        # 计算重叠度 (使用最小值方法)
        overlap = np.sum(np.minimum(hist_in_norm, hist_out_norm))
        
        return float(overlap)
    
    def get_all_metrics(self, lin: np.ndarray, lout: np.ndarray) -> Dict[str, Union[float, str]]:
        """
        一次性计算所有质量指标
        性能优化：确保30ms内完成1MP图像处理
        
        Args:
            lin: 输入亮度数据 (PQ域，范围0-1)
            lout: 输出亮度数据 (PQ域，映射后，范围0-1)
            
        Returns:
            包含所有指标的字典，数值格式为小数
        """
        import time
        start_time = time.perf_counter()  # 使用更精确的计时器
        
        try:
            # 验证输入
            if lin is None or lout is None:
                raise ValueError("输入数据不能为None")
            
            # 性能优化：使用float32减少内存占用和计算时间
            # 如果输入已经是正确类型，避免不必要的转换
            if not isinstance(lin, np.ndarray) or lin.dtype != np.float32:
                lin_array = np.asarray(lin, dtype=np.float32)
            else:
                lin_array = lin
                
            if not isinstance(lout, np.ndarray) or lout.dtype != np.float32:
                lout_array = np.asarray(lout, dtype=np.float32)
            else:
                lout_array = lout
            
            if lin_array.shape != lout_array.shape:
                raise ValueError(f"输入输出形状不匹配: {lin_array.shape} vs {lout_array.shape}")
            
            if lin_array.size == 0:
                raise ValueError("输入数组为空")
            
            # 性能优化：预先展平数组，避免重复操作
            # 使用ravel()而不是flatten()，ravel()返回视图而不是副本（如果可能）
            lin_flat = lin_array.ravel()
            lout_flat = lout_array.ravel()
            
            # 性能优化：一次性计算所有需要的统计量，避免多次遍历数组
            metrics = self._calculate_all_metrics_vectorized(lin_flat, lout_flat)
            
            # 自动质量评估（使用最新阈值配置）
            exposure_status = self.evaluate_quality_status(metrics)
            metrics['Exposure_status'] = exposure_status
            
            # 添加格式化的状态显示信息
            metrics['Status_display'] = self.get_formatted_status_display(exposure_status)
            
            # 性能监控
            elapsed_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
            
            # 记录性能统计
            if self.performance_monitor:
                self.performance_monitor.record_assessment(elapsed_time)
            
            if elapsed_time > 30:
                self.logger.warning(f"质量指标计算耗时 {elapsed_time:.1f}ms，超过30ms目标")
            else:
                self.logger.debug(f"质量指标计算耗时 {elapsed_time:.1f}ms")
            
            # 内存管理：对于大数组处理后进行垃圾回收
            if lin_array.size > self.large_array_threshold:
                gc.collect()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算质量指标时发生错误: {e}")
            return {
                'error': str(e),
                'Exposure_status': '计算失败'
            }
    
    def _calculate_exposure_metrics_optimized(self, lin_flat: np.ndarray, lout_flat: np.ndarray) -> Dict[str, float]:
        """
        优化版曝光指标计算
        
        Args:
            lin_flat: 展平的输入亮度数据
            lout_flat: 展平的输出亮度数据
            
        Returns:
            曝光指标字典
        """
        total_pixels = len(lout_flat)
        
        # 批量计算所有阈值掩码
        highlight_mask = lout_flat > 0.9
        shadow_mask_out = lout_flat < 0.05  # 使用0.05作为暗部阈值
        
        # S_ratio (高光饱和比例)
        s_ratio = np.sum(highlight_mask) / total_pixels if total_pixels > 0 else 0.0
        
        # C_shadow (暗部压缩比例) - 简化为输出暗部像素比例
        c_shadow = np.sum(shadow_mask_out) / total_pixels if total_pixels > 0 else 0.0
        
        # R_DR (动态范围保持率) - 向量化计算
        dr_in = np.ptp(lin_flat)  # ptp = max - min，更高效
        dr_out = np.ptp(lout_flat)
        
        r_dr = dr_out / dr_in if dr_in > self.eps else 1.0
        
        # ΔL_mean_norm (归一化平均亮度漂移) - 向量化计算
        # 按照草稿要求：亮度差值除以动态范围
        mean_in = np.mean(lin_flat)
        mean_out = np.mean(lout_flat)
        dr_in = np.max(lin_flat) - np.min(lin_flat)
        
        delta_l_mean_norm = abs(mean_out - mean_in) / dr_in if dr_in > self.eps else 0.0
        
        return {
            'S_ratio': float(s_ratio),
            'C_shadow': float(c_shadow),
            'R_DR': float(r_dr),
            'ΔL_mean_norm': float(delta_l_mean_norm)
        }
    
    def _calculate_histogram_overlap_optimized(self, lin_flat: np.ndarray, lout_flat: np.ndarray) -> float:
        """
        优化版直方图重叠度计算
        
        Args:
            lin_flat: 展平的输入亮度数据
            lout_flat: 展平的输出亮度数据
            
        Returns:
            直方图重叠度
        """
        # 使用固定bins和range提高性能
        bins = 256
        range_pq = (0.0, 1.0)
        
        # 批量计算两个直方图
        hist_in, _ = np.histogram(lin_flat, bins=bins, range=range_pq, density=True)
        hist_out, _ = np.histogram(lout_flat, bins=bins, range=range_pq, density=True)
        
        # 向量化归一化
        sum_in = np.sum(hist_in)
        sum_out = np.sum(hist_out)
        
        if sum_in > self.eps and sum_out > self.eps:
            hist_in_norm = hist_in / sum_in
            hist_out_norm = hist_out / sum_out
            
            # 按照草稿要求：1 - 0.5 * Σ|h_in - h_out|
            overlap = 1.0 - 0.5 * np.sum(np.abs(hist_in_norm - hist_out_norm))
        else:
            overlap = 0.0
        
        return float(overlap)
    
    def _calculate_all_metrics_vectorized(self, lin_flat: np.ndarray, lout_flat: np.ndarray) -> Dict[str, float]:
        """
        高度优化的向量化指标计算
        一次性计算所有指标，最大化性能
        
        Args:
            lin_flat: 展平的输入亮度数据
            lout_flat: 展平的输出亮度数据
            
        Returns:
            所有质量指标的字典
        """
        total_pixels = len(lout_flat)
        
        # 一次性计算所有需要的统计量
        lin_min, lin_max = np.min(lin_flat), np.max(lin_flat)
        lout_min, lout_max = np.min(lout_flat), np.max(lout_flat)
        lin_mean = np.mean(lin_flat)
        lout_mean = np.mean(lout_flat)
        
        # 批量计算所有阈值掩码
        highlight_mask = lout_flat > 0.9
        shadow_mask_out = lout_flat < 0.05  # 使用0.05作为暗部阈值
        
        # 向量化计算所有指标
        # S_ratio (高光饱和比例)
        s_ratio = np.sum(highlight_mask) / total_pixels if total_pixels > 0 else 0.0
        
        # C_shadow (暗部压缩比例) - 简化为输出暗部像素比例
        c_shadow = np.sum(shadow_mask_out) / total_pixels if total_pixels > 0 else 0.0
        
        # R_DR (动态范围保持率) - 使用预计算的min/max
        dr_in = lin_max - lin_min
        dr_out = lout_max - lout_min
        r_dr = dr_out / dr_in if dr_in > self.eps else 1.0
        
        # ΔL_mean_norm (归一化平均亮度漂移) - 使用预计算的mean
        # 按照草稿要求：亮度差值除以动态范围
        dr_in = lin_max - lin_min
        delta_l_mean_norm = abs(lout_mean - lin_mean) / dr_in if dr_in > self.eps else 0.0
        
        # 直方图重叠度 - 优化版本
        hist_overlap = self._calculate_histogram_overlap_optimized(lin_flat, lout_flat)
        
        return {
            # 基础统计数据
            'Lmin_in': float(lin_min),
            'Lmax_in': float(lin_max),
            'Lmin_out': float(lout_min),
            'Lmax_out': float(lout_max),
            
            # 曝光相关指标
            'S_ratio': float(s_ratio),
            'C_shadow': float(c_shadow),
            'R_DR': float(r_dr),
            'ΔL_mean_norm': float(delta_l_mean_norm),
            
            # 直方图分析
            'Hist_overlap': hist_overlap
        }
    
    def to_json(self, metrics: Dict[str, Union[float, str]], indent: int = 2) -> str:
        """
        将指标转换为JSON格式字符串
        确保数值格式为小数
        
        Args:
            metrics: 指标字典
            indent: JSON缩进
            
        Returns:
            JSON格式字符串
        """
        try:
            # 确保所有数值都是标准Python类型，便于JSON序列化
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                else:
                    json_metrics[key] = value
            
            return json.dumps(json_metrics, indent=indent, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"JSON序列化失败: {e}")
            return json.dumps({"error": str(e)}, indent=indent)
    
    def evaluate_quality_status(self, metrics: Dict[str, float]) -> str:
        """
        基于指标自动判断质量状态
        实现基于阈值的状态判断逻辑(过曝、过暗、动态范围异常、正常)
        
        Args:
            metrics: 质量指标字典
            
        Returns:
            质量状态字符串
        """
        try:
            # 确保使用最新的阈值配置（支持热更新）
            current_thresholds = self.config_manager.load_thresholds()
            
            s_ratio = metrics.get('S_ratio', 0.0)
            c_shadow = metrics.get('C_shadow', 0.0)
            r_dr = metrics.get('R_DR', 1.0)
            
            # 计算Dprime (简化版感知失真)
            delta_l_mean_norm = metrics.get('ΔL_mean_norm', 1.0)
            dprime = abs(delta_l_mean_norm - 1.0)
            
            # 按照需求2的验收标准进行状态判断
            
            # 1. 判断过曝：当S_ratio大于阈值或Dprime大于阈值时
            if s_ratio > current_thresholds['S_ratio'] or dprime > current_thresholds['Dprime']:
                return "过曝"
            
            # 2. 判断过暗：当C_shadow大于阈值时
            if c_shadow > current_thresholds['C_shadow']:
                return "过暗"
            
            # 3. 判断动态范围异常：当R_DR偏离1.0超过阈值时
            r_dr_deviation = abs(r_dr - 1.0)
            if r_dr_deviation > current_thresholds['R_DR_tolerance']:
                return "动态范围异常"
            
            # 4. 正常状态：当所有指标都在正常范围内时
            return "正常"
            
        except Exception as e:
            self.logger.error(f"评估质量状态时发生错误: {e}")
            return "评估失败"
    
    def get_status_display_info(self, status: str) -> Dict[str, str]:
        """
        获取状态的颜色编码和文本标识信息
        开发状态颜色编码和文本标识系统
        
        Args:
            status: 质量状态字符串
            
        Returns:
            包含颜色编码、emoji和显示文本的字典
        """
        status_mapping = {
            "正常": {
                "color": "green",
                "emoji": "🟢",
                "text": "正常",
                "description": "图像质量良好，各项指标均在正常范围内"
            },
            "过曝": {
                "color": "red", 
                "emoji": "🔴",
                "text": "过曝",
                "description": "图像存在过曝问题，高光区域饱和或亮度漂移过大"
            },
            "过暗": {
                "color": "purple",
                "emoji": "🟣", 
                "text": "暗压",
                "description": "图像暗部压缩过度，细节丢失"
            },
            "动态范围异常": {
                "color": "white",
                "emoji": "⚪",
                "text": "异常",
                "description": "动态范围保持率异常，映射效果不理想"
            },
            "评估失败": {
                "color": "gray",
                "emoji": "⚫",
                "text": "失败",
                "description": "质量评估过程中发生错误"
            }
        }
        
        return status_mapping.get(status, {
            "color": "gray",
            "emoji": "❓",
            "text": "未知",
            "description": f"未知状态: {status}"
        })
    
    def get_formatted_status_display(self, status: str) -> str:
        """
        获取格式化的状态显示文本（包含emoji和颜色编码）
        
        Args:
            status: 质量状态字符串
            
        Returns:
            格式化的状态显示文本
        """
        display_info = self.get_status_display_info(status)
        return f"{display_info['emoji']} {display_info['text']}"
    
    def get_performance_report(self) -> Optional[Dict[str, Union[float, int, str]]]:
        """
        获取性能监控报告
        
        Returns:
            性能报告字典，如果未启用监控则返回None
        """
        if self.performance_monitor:
            return self.performance_monitor.get_performance_report()
        return None
    
    def reset_performance_stats(self) -> None:
        """重置性能统计"""
        if self.performance_monitor:
            self.performance_monitor.stats = {
                'total_assessments': 0,
                'total_time_ms': 0.0,
                'max_time_ms': 0.0,
                'min_time_ms': float('inf'),
                'over_target_count': 0
            }
    
    def optimize_for_large_images(self, enable: bool = True) -> None:
        """
        为大图像处理优化设置
        
        Args:
            enable: 是否启用大图像优化
        """
        if enable:
            # 调整垃圾回收阈值
            gc.set_threshold(500, 5, 5)
            # 降低大数组阈值
            self.large_array_threshold = 500000  # 0.5MP
            self.logger.info("已启用大图像处理优化")
        else:
            # 恢复默认设置
            gc.set_threshold(700, 10, 10)
            self.large_array_threshold = 1000000  # 1MP
            self.logger.info("已禁用大图像处理优化")