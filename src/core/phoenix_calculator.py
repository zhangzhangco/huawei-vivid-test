"""
Phoenix曲线计算器
实现Phoenix色调映射曲线计算，包含数值稳定性和单调性验证
"""

import numpy as np
from typing import Tuple, Union, Optional, Any
import logging


class PhoenixCurveCalculator:
    """Phoenix曲线计算器"""
    
    def __init__(self, display_samples: int = 512, validation_samples: int = 1024,
                 sampling_optimizer: Optional[Any] = None):
        # 数值稳定性参数
        self.eps = 1e-6

        # 采样参数
        self.display_samples = display_samples
        self.validation_samples = validation_samples

        # 可选的采样优化器
        self._sampling_optimizer = sampling_optimizer

        # 参数范围
        self.p_range = (0.1, 6.0)
        self.a_range = (0.0, 1.0)
        
    def compute_phoenix_curve(self, L: Union[np.ndarray, float], p: float, a: float) -> Union[np.ndarray, float]:
        """
        计算Phoenix曲线: L' = L^p / (L^p + a^p)
        
        Args:
            L: 输入亮度数组 (PQ域, 0-1)
            p: 亮度控制因子 (0.1-6.0)
            a: 缩放因子 (0.0-1.0)
            
        Returns:
            L_out: 输出亮度数组
            
        Raises:
            ValueError: 参数超出有效范围
        """
        # 参数验证
        if not (self.p_range[0] <= p <= self.p_range[1]):
            raise ValueError(f"参数p={p}超出范围{self.p_range}")
        if not (self.a_range[0] <= a <= self.a_range[1]):
            raise ValueError(f"参数a={a}超出范围{self.a_range}")
            
        # 输入夹取和安全处理
        L_array = np.asarray(L)
        L_clipped = np.clip(L_array, self.eps, 1.0)

        # Phoenix曲线计算（数值稳定，无需特殊处理a=0的情况）
        L_p = np.power(L_clipped, p)
        a_p = np.power(a, p)

        return L_p / (L_p + a_p)
        
    def normalize_endpoints(self, L_out: np.ndarray, L_min: float = 0.0, 
                          L_max: float = 1.0) -> np.ndarray:
        """
        端点归一化到显示设备范围
        确保端点严格匹配：L=0→L'=L_min，L=1→L'=L_max

        注意：此操作会线性拉伸曲线以对齐端点，会改变原始曲线形状。
        通过线性重映射将 Phoenix 曲线的输出范围映射到显示设备的PQ范围。

        Args:
            L_out: Phoenix曲线输出
            L_min: 显示设备最小PQ值
            L_max: 显示设备最大PQ值

        Returns:
            线性重映射后的曲线
        """
        if len(L_out) < 2:
            return L_out
            
        eps = 1e-8
        L0, L1 = float(L_out[0]), float(L_out[-1])
        denom = max(abs(L1 - L0), eps)
        
        # 线性归一化
        s = (L_out - L0) / denom
        normalized = L_min + (L_max - L_min) * s
        
        return np.clip(normalized, L_min, L_max)
        
    def validate_monotonicity(self, L_out: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        验证曲线单调性
        
        Args:
            L_out: 输出曲线
            tolerance: 数值容差
            
        Returns:
            是否单调递增
        """
        if len(L_out) < 2:
            return True
            
        diff = np.diff(L_out)
        return bool(np.all(diff >= -tolerance))
        
    def validate_monotonicity_pa(self, p: float, a: float, use_optimized_sampling: bool = True) -> bool:
        """
        基于高密度采样验证参数组合的单调性
        
        Args:
            p: 亮度控制因子
            a: 缩放因子
            use_optimized_sampling: 是否使用优化的采样密度
            
        Returns:
            是否单调递增
        """
        try:
            if use_optimized_sampling and self._sampling_optimizer is not None:
                samples = self._sampling_optimizer.optimize_sampling_density("validation")
            else:
                samples = self.validation_samples
                
            L_validation = np.linspace(0, 1, samples)
            L_out_validation = self.compute_phoenix_curve(L_validation, p, a)
            return self.validate_monotonicity(L_out_validation)
        except Exception as e:
            logging.warning(f"单调性验证失败: {e}")
            return False
            
    def get_display_curve(self, p: float, a: float, use_optimized_sampling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取用于UI显示的曲线
        
        Args:
            p: 亮度控制因子
            a: 缩放因子
            use_optimized_sampling: 是否使用优化的采样密度
            
        Returns:
            (输入亮度数组, 输出亮度数组)
        """
        if use_optimized_sampling and self._sampling_optimizer is not None:
            samples = self._sampling_optimizer.optimize_sampling_density("display")
        else:
            samples = self.display_samples
            
        L = np.linspace(0, 1, samples)
        L_out = self.compute_phoenix_curve(L, p, a)
        return L, L_out
        
    def get_validation_curve(self, p: float, a: float, use_optimized_sampling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取用于单调性验证的高密度曲线
        
        Args:
            p: 亮度控制因子
            a: 缩放因子
            use_optimized_sampling: 是否使用优化的采样密度
            
        Returns:
            (输入亮度数组, 输出亮度数组)
        """
        if use_optimized_sampling and self._sampling_optimizer is not None:
            samples = self._sampling_optimizer.optimize_sampling_density("validation")
        else:
            samples = self.validation_samples
            
        L = np.linspace(0, 1, samples)
        L_out = self.compute_phoenix_curve(L, p, a)
        return L, L_out
        
    def compute_curve_derivative(self, L: np.ndarray, p: float, a: float) -> np.ndarray:
        """
        计算Phoenix曲线的导数 (用于单调性分析)
        
        Args:
            L: 输入亮度数组
            p: 亮度控制因子
            a: 缩放因子
            
        Returns:
            曲线导数
        """
        L_clipped = np.clip(L, self.eps, 1.0)
        a_eff = max(a, self.eps)
        
        L_p = np.power(L_clipped, p)
        a_p = np.power(a_eff, p)
        
        # dL'/dL = p * L^(p-1) * a^p / (L^p + a^p)^2
        numerator = p * np.power(L_clipped, p - 1) * a_p
        denominator = np.power(L_p + a_p, 2)
        
        return numerator / np.maximum(denominator, self.eps)
        
    def check_endpoint_accuracy(self, L_out: np.ndarray, L_min: float = 0.0, 
                               L_max: float = 1.0) -> float:
        """
        检查端点匹配精度
        
        Args:
            L_out: 输出曲线
            L_min: 期望的最小值
            L_max: 期望的最大值
            
        Returns:
            端点误差 (最大绝对误差)
        """
        if len(L_out) < 2:
            return 0.0
            
        error_min = abs(L_out[0] - L_min)
        error_max = abs(L_out[-1] - L_max)
        
        return max(error_min, error_max)