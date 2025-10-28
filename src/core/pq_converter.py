"""
ST 2084 (PQ) 转换器
实现线性光与PQ域之间的转换，支持sRGB到线性光的转换
"""

import numpy as np
from typing import Union


class PQConverter:
    """ST 2084 (PQ) 转换器"""
    
    def __init__(self):
        # ST 2084 常数 (SMPTE ST 2084 / BT.2100)
        self.m1 = 2610.0 / 16384.0
        self.m2 = 2523.0 / 4096.0 * 128.0
        self.c1 = 3424.0 / 4096.0
        self.c2 = 2413.0 / 4096.0 * 32.0
        self.c3 = 2392.0 / 4096.0 * 32.0
        
        # 数值稳定性参数
        self.eps = 1e-10
        
    def linear_to_pq(self, linear: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        线性光 -> PQ域转换
        
        Args:
            linear: 线性光亮度值 (nits)
            
        Returns:
            PQ域值 (0-1范围)
        """
        # 归一化到10000 nits
        Y = np.clip(np.asarray(linear) / 10000.0, 0, 1)
        Y_m1 = np.power(Y, self.m1)
        numerator = self.c1 + self.c2 * Y_m1
        denominator = 1 + self.c3 * Y_m1
        
        # 避免除零
        denominator = np.maximum(denominator, self.eps)
        
        result = np.power(numerator / denominator, self.m2)
        return np.clip(result, 0, 1)
        
    def pq_to_linear(self, pq: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        PQ域 -> 线性光转换
        
        Args:
            pq: PQ域值 (0-1范围)
            
        Returns:
            线性光亮度值 (nits)
        """
        pq_clipped = np.clip(np.asarray(pq), 0, 1)
        pq_m2_inv = np.power(pq_clipped, 1.0 / self.m2)
        numerator = np.maximum(pq_m2_inv - self.c1, 0)
        denominator = self.c2 - self.c3 * pq_m2_inv
        
        # 避免除零
        denominator = np.maximum(denominator, self.eps)
        
        Y = np.power(numerator / denominator, 1.0 / self.m1)
        return Y * 10000.0
        
    def srgb_to_linear(self, srgb: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        sRGB -> 线性光转换
        
        Args:
            srgb: sRGB值 (0-1范围)
            
        Returns:
            线性光值 (0-1范围，相对亮度)
        """
        srgb_clipped = np.clip(np.asarray(srgb), 0, 1)
        return np.where(srgb_clipped <= 0.04045, 
                       srgb_clipped / 12.92,
                       np.power((srgb_clipped + 0.055) / 1.055, 2.4))
                       
    def linear_to_srgb(self, linear: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        线性光 -> sRGB转换
        
        Args:
            linear: 线性光值 (0-1范围，相对亮度)
            
        Returns:
            sRGB值 (0-1范围)
        """
        linear_clipped = np.clip(np.asarray(linear), 0, 1)
        return np.where(linear_clipped <= 0.0031308,
                       linear_clipped * 12.92,
                       1.055 * np.power(linear_clipped, 1.0 / 2.4) - 0.055)