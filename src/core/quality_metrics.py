"""
质量指标计算器
实现感知失真D'、局部对比度计算、带滞回特性的模式推荐系统
支持MaxRGB和Y通道亮度提取，以及PQ域直方图计算
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import logging


class QualityMetricsCalculator:
    """质量指标计算器"""
    
    def __init__(self, luminance_channel: str = "MaxRGB"):
        """
        初始化质量指标计算器
        
        Args:
            luminance_channel: 亮度通道类型 ("MaxRGB" 或 "Y")
        """
        self.eps = 1e-8
        self.luminance_channel = luminance_channel
        
        # 滞回阈值 (需求 10.1, 10.4)
        self.dt_low = 0.05   # 滞回下阈值
        self.dt_high = 0.10  # 滞回上阈值
        self.last_recommendation = None
        
        # BT.2100 Y通道权重
        self.y_weights = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)
        
    def set_luminance_channel(self, channel: str):
        """
        设置亮度通道类型
        
        Args:
            channel: "MaxRGB" 或 "Y"
        """
        if channel not in ["MaxRGB", "Y"]:
            raise ValueError(f"不支持的亮度通道: {channel}")
        self.luminance_channel = channel
        
    def extract_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        提取亮度通道 (需求 12.1)
        使用共享的 extract_luminance 函数确保一致性

        Args:
            image: 输入图像 (H, W), (H, W, 1), (H, W, 3) 或 (H, W, 4)

        Returns:
            亮度通道数组 (float32)
        """
        from .image_processor import extract_luminance
        return extract_luminance(image, self.luminance_channel).astype(np.float32)
            
    def compute_perceptual_distortion(self, L_in: np.ndarray, L_out: np.ndarray) -> float:
        """
        计算平均亮度差 (需求 3.1)
        D' = |mean(L_out) - mean(L_in)|

        注意：此函数实际计算的是平均亮度差，并非复杂的感知失真模型。
        这是一个简单指标，用于快速评估色调映射前后的整体亮度变化。
        后续可考虑引入更真实的感知模型（如SSIM、VMAF等）进行扩展。

        Args:
            L_in: 输入亮度数组 (PQ域)
            L_out: 输出亮度数组 (PQ域)

        Returns:
            平均亮度差值
        """
        L_in_array = np.asarray(L_in, dtype=np.float32)
        L_out_array = np.asarray(L_out, dtype=np.float32)

        mean_in = np.mean(L_in_array)
        mean_out = np.mean(L_out_array)

        return float(abs(mean_out - mean_in))
        
    def compute_local_contrast(self, L: np.ndarray) -> float:
        """
        计算局部对比度 (需求 3.2)
        局部对比度 = mean(|∇L|) (相邻像素亮度梯度的平均值)

        注意：使用 np.diff 的轻量近似实现，适合实时UI场景。
        相比 Sobel/Scharr 等复杂梯度算子，计算量更低但精度有限。
        未来可作为可选模式扩展更精确的梯度算法。

        Args:
            L: 亮度数组 (PQ域)

        Returns:
            局部对比度值
        """
        L_array = np.asarray(L, dtype=np.float32)

        if L_array.ndim == 1:
            # 1D数组：计算相邻差分
            if len(L_array) < 2:
                return 0.0
            diff = np.abs(np.diff(L_array))
            return float(np.mean(diff))
        elif L_array.ndim == 2:
            # 2D图像：计算梯度幅值
            grad_x = np.abs(np.diff(L_array, axis=1))
            grad_y = np.abs(np.diff(L_array, axis=0))

            # 计算平均梯度幅值
            contrast_x = np.mean(grad_x) if grad_x.size > 0 else 0.0
            contrast_y = np.mean(grad_y) if grad_y.size > 0 else 0.0

            return float((contrast_x + contrast_y) / 2.0)
        else:
            raise ValueError(f"不支持的数组维度: {L_array.shape}")
            
    def compute_variance_distortion(self, L_in: np.ndarray, L_out: np.ndarray) -> float:
        """
        计算方差失真 (备选指标)
        
        Args:
            L_in: 输入亮度数组
            L_out: 输出亮度数组
            
        Returns:
            方差失真值
        """
        L_in_array = np.asarray(L_in, dtype=np.float64)
        L_out_array = np.asarray(L_out, dtype=np.float64)
        
        var_in = np.var(L_in_array)
        var_out = np.var(L_out_array)
        
        return float(abs(var_out - var_in) / (var_in + self.eps))
        
    def recommend_mode_with_hysteresis(self, distortion: float) -> str:
        """
        带滞回特性的模式推荐 (需求 10.1, 10.4)
        
        Args:
            distortion: 感知失真值
            
        Returns:
            推荐模式 ("自动模式" 或 "艺术模式")
        """
        if distortion <= self.dt_low:
            self.last_recommendation = "自动模式"
        elif distortion >= self.dt_high:
            self.last_recommendation = "艺术模式"
        # 中间区间 [dt_low, dt_high] 保持上次决策
        
        return self.last_recommendation or "艺术模式"
        
    def compute_histogram(self, L: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算PQ域直方图 (需求 16.2)
        
        Args:
            L: 亮度数组 (PQ域, 0-1)
            bins: 直方图bin数量
            
        Returns:
            (直方图计数, bin边界)
        """
        L_array = np.asarray(L, dtype=np.float64)
        L_clipped = np.clip(L_array.flatten(), 0.0, 1.0)
        
        hist, bin_edges = np.histogram(L_clipped, bins=bins, range=(0.0, 1.0))
        
        return hist.astype(np.int32), bin_edges.astype(np.float64)
        
    def compute_histogram_stats(self, L: np.ndarray) -> Dict[str, float]:
        """
        计算直方图统计信息
        
        Args:
            L: 亮度数组 (PQ域)
            
        Returns:
            统计信息字典
        """
        L_array = np.asarray(L, dtype=np.float64)
        L_clipped = np.clip(L_array.flatten(), 0.0, 1.0)
        
        return {
            'min_pq': float(np.min(L_clipped)),
            'max_pq': float(np.max(L_clipped)),
            'avg_pq': float(np.mean(L_clipped)),
            'var_pq': float(np.var(L_clipped)),
            'std_pq': float(np.std(L_clipped)),
            'median_pq': float(np.median(L_clipped)),
            'percentile_5': float(np.percentile(L_clipped, 5)),
            'percentile_95': float(np.percentile(L_clipped, 95))
        }
        
    def compute_all_metrics(self, L_in: np.ndarray, L_out: np.ndarray) -> Dict[str, float]:
        """
        计算所有质量指标
        
        Args:
            L_in: 输入亮度数组
            L_out: 输出亮度数组
            
        Returns:
            所有指标的字典
        """
        metrics = {}
        
        # 基本质量指标
        metrics['perceptual_distortion'] = self.compute_perceptual_distortion(L_in, L_out)
        metrics['local_contrast_in'] = self.compute_local_contrast(L_in)
        metrics['local_contrast_out'] = self.compute_local_contrast(L_out)
        metrics['variance_distortion'] = self.compute_variance_distortion(L_in, L_out)
        
        # 模式推荐
        metrics['recommended_mode'] = self.recommend_mode_with_hysteresis(
            metrics['perceptual_distortion']
        )
        
        # 统计信息
        stats_in = self.compute_histogram_stats(L_in)
        stats_out = self.compute_histogram_stats(L_out)
        
        for key, value in stats_in.items():
            metrics[f'input_{key}'] = value
        for key, value in stats_out.items():
            metrics[f'output_{key}'] = value
            
        # 元信息
        metrics['luminance_channel'] = self.luminance_channel
        metrics['dt_low'] = self.dt_low
        metrics['dt_high'] = self.dt_high
        
        return metrics
        
    def reset_hysteresis(self):
        """重置滞回状态"""
        self.last_recommendation = None
        
    def set_hysteresis_thresholds(self, dt_low: float, dt_high: float):
        """
        设置滞回阈值
        
        Args:
            dt_low: 下阈值
            dt_high: 上阈值
        """
        if dt_low >= dt_high:
            raise ValueError(f"下阈值({dt_low})必须小于上阈值({dt_high})")
        if not (0.0 <= dt_low <= 1.0) or not (0.0 <= dt_high <= 1.0):
            raise ValueError("阈值必须在[0,1]范围内")
            
        self.dt_low = dt_low
        self.dt_high = dt_high
        
    def validate_inputs(self, L_in: np.ndarray, L_out: np.ndarray) -> Tuple[bool, str]:
        """
        验证输入数据
        
        Args:
            L_in: 输入亮度数组
            L_out: 输出亮度数组
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            L_in_array = np.asarray(L_in)
            L_out_array = np.asarray(L_out)
            
            if L_in_array.shape != L_out_array.shape:
                return False, f"输入输出形状不匹配: {L_in_array.shape} vs {L_out_array.shape}"
                
            if L_in_array.size == 0:
                return False, "输入数组为空"
                
            if not np.all(np.isfinite(L_in_array)) or not np.all(np.isfinite(L_out_array)):
                return False, "输入包含非有限值(NaN或Inf)"
                
            return True, ""
            
        except Exception as e:
            return False, f"输入验证失败: {str(e)}"


class ImageQualityAnalyzer:
    """图像质量分析器 - 专门处理图像相关的质量分析"""
    
    def __init__(self, luminance_channel: str = "MaxRGB"):
        self.metrics_calculator = QualityMetricsCalculator(luminance_channel)
        
    def analyze_image_quality(self, image_in: np.ndarray, image_out: np.ndarray) -> Dict[str, Union[float, str, np.ndarray]]:
        """
        分析图像质量
        
        Args:
            image_in: 输入图像
            image_out: 输出图像
            
        Returns:
            质量分析结果
        """
        # 提取亮度通道
        L_in = self.metrics_calculator.extract_luminance(image_in)
        L_out = self.metrics_calculator.extract_luminance(image_out)
        
        # 验证输入
        valid, msg = self.metrics_calculator.validate_inputs(L_in, L_out)
        if not valid:
            return {'error': msg}
            
        # 计算所有指标
        metrics = self.metrics_calculator.compute_all_metrics(L_in, L_out)
        
        # 计算直方图
        hist_in, bin_edges = self.metrics_calculator.compute_histogram(L_in)
        hist_out, _ = self.metrics_calculator.compute_histogram(L_out)
        
        metrics['histogram_in'] = hist_in
        metrics['histogram_out'] = hist_out
        metrics['histogram_bins'] = bin_edges
        
        return metrics
        
    def compare_tone_mapping_results(self, original: np.ndarray, 
                                   results: List[Tuple[str, np.ndarray]]) -> Dict[str, Dict]:
        """
        比较多个色调映射结果
        
        Args:
            original: 原始图像
            results: [(方法名, 结果图像), ...]
            
        Returns:
            比较结果
        """
        L_original = self.metrics_calculator.extract_luminance(original)
        comparison = {}
        
        for method_name, result_image in results:
            try:
                L_result = self.metrics_calculator.extract_luminance(result_image)
                metrics = self.metrics_calculator.compute_all_metrics(L_original, L_result)
                comparison[method_name] = metrics
            except Exception as e:
                comparison[method_name] = {'error': str(e)}
                
        return comparison