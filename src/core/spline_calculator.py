"""
样条曲线计算器
实现分段三次Hermite样条(PCHIP)计算，支持C¹连续性验证和单调性检查
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy.interpolate import PchipInterpolator
import logging

from .parameter_validator import ParameterValidator


class SplineCalculationError(Exception):
    """样条曲线计算错误"""
    pass


class SplineCurveCalculator:
    """样条曲线计算器
    
    实现分段三次Hermite样条(PCHIP)计算，确保C¹连续性和单调性
    """
    
    def __init__(self):
        self.default_nodes = [0.2, 0.5, 0.8]
        self.min_node_interval = 0.01
        self.eps = 1e-6
        self.continuity_tolerance = 1e-3
        
    def validate_and_correct_nodes(self, nodes: List[float]) -> Tuple[List[float], str]:
        """验证并修正样条节点
        
        Args:
            nodes: 样条节点列表
            
        Returns:
            corrected_nodes: 修正后的节点
            warning: 警告信息
        """
        return ParameterValidator.validate_spline_nodes(nodes, self.min_node_interval)
        
    def compute_pchip_spline(self, x_nodes: np.ndarray, y_nodes: np.ndarray, 
                           x_eval: np.ndarray) -> np.ndarray:
        """计算PCHIP样条插值
        
        Args:
            x_nodes: 节点x坐标
            y_nodes: 节点y坐标  
            x_eval: 评估点x坐标
            
        Returns:
            y_eval: 插值结果
        """
        try:
            # 使用scipy的PCHIP插值器，自动保证单调性
            interpolator = PchipInterpolator(x_nodes, y_nodes)
            return interpolator(x_eval)
        except Exception as e:
            raise SplineCalculationError(f"PCHIP插值计算失败: {e}")
            
    def verify_c1_continuity(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> Tuple[bool, float]:
        """验证C¹连续性
        
        Args:
            x_nodes: 节点x坐标
            y_nodes: 节点y坐标
            
        Returns:
            is_continuous: 是否满足C¹连续性
            max_error: 最大连续性误差
        """
        try:
            interpolator = PchipInterpolator(x_nodes, y_nodes)
            
            # 检查内部节点的导数连续性
            max_error = 0.0
            for i in range(1, len(x_nodes) - 1):
                x = x_nodes[i]
                # 计算左导数和右导数
                dx = 1e-8
                left_deriv = (interpolator(x) - interpolator(x - dx)) / dx
                right_deriv = (interpolator(x + dx) - interpolator(x)) / dx
                
                error = abs(right_deriv - left_deriv)
                max_error = max(max_error, error)
                
            is_continuous = max_error <= self.continuity_tolerance
            return is_continuous, max_error
            
        except Exception as e:
            logging.warning(f"C¹连续性验证失败: {e}")
            return False, float('inf')
            
    def create_spline_from_phoenix(self, phoenix_curve: np.ndarray, 
                                 x_input: np.ndarray,
                                 th_nodes: List[float]) -> Tuple[np.ndarray, bool, str]:
        """基于Phoenix曲线创建样条曲线
        
        Args:
            phoenix_curve: Phoenix曲线输出
            x_input: 输入x坐标
            th_nodes: 样条节点位置
            
        Returns:
            spline_curve: 样条曲线输出
            success: 是否成功
            message: 状态信息
        """
        try:
            # 验证和修正节点
            corrected_nodes, warning = self.validate_and_correct_nodes(th_nodes)
            
            # 构建样条节点：包含端点和中间节点
            x_nodes = np.array([0.0] + corrected_nodes + [1.0])
            
            # 在节点位置插值Phoenix曲线值，确保端点锚定
            y_nodes = np.interp(x_nodes, x_input, phoenix_curve)
            
            # 计算PCHIP样条
            spline_curve = self.compute_pchip_spline(x_nodes, y_nodes, x_input)
            
            # 验证C¹连续性
            is_continuous, continuity_error = self.verify_c1_continuity(x_nodes, y_nodes)
            
            message = warning
            if not is_continuous:
                message += f" C¹连续性误差: {continuity_error:.6f}"
                
            return spline_curve, True, message
            
        except Exception as e:
            logging.error(f"样条曲线创建失败: {e}")
            return phoenix_curve.copy(), False, f"样条计算错误: {str(e)}"
            
    def blend_with_phoenix(self, phoenix_curve: np.ndarray, 
                          spline_curve: np.ndarray, 
                          strength: float) -> np.ndarray:
        """Phoenix曲线与样条曲线的凸组合
        
        Args:
            phoenix_curve: Phoenix曲线
            spline_curve: 样条曲线
            strength: 样条强度 (0-1)
            
        Returns:
            blended_curve: 混合后的曲线
        """
        # 确保strength在有效范围内
        strength = np.clip(strength, 0.0, 1.0)
        
        # 凸组合公式: L'_final = (1-strength)*L'_phoenix + strength*L'_spline
        return (1.0 - strength) * phoenix_curve + strength * spline_curve
        
    def check_monotonicity(self, curve: np.ndarray) -> bool:
        """检查曲线单调性
        
        Args:
            curve: 待检查的曲线
            
        Returns:
            is_monotonic: 是否单调递增
        """
        return bool(np.all(np.diff(curve) >= -self.eps))  # 允许微小的数值误差
        
    def compute_spline_with_fallback(self, phoenix_curve: np.ndarray,
                                   x_input: np.ndarray,
                                   th_nodes: List[float],
                                   th_strength: float) -> Tuple[np.ndarray, bool, str]:
        """计算样条曲线，支持自动回退
        
        Args:
            phoenix_curve: Phoenix曲线
            x_input: 输入x坐标
            th_nodes: 样条节点
            th_strength: 样条强度
            
        Returns:
            final_curve: 最终曲线
            used_spline: 是否使用了样条
            status_message: 状态信息
        """
        # 如果强度为0，直接返回Phoenix曲线
        if th_strength <= self.eps:
            return phoenix_curve.copy(), False, "样条强度为0，使用Phoenix曲线"
            
        # 尝试创建样条曲线
        spline_curve, spline_success, spline_message = self.create_spline_from_phoenix(
            phoenix_curve, x_input, th_nodes
        )
        
        if not spline_success:
            return phoenix_curve.copy(), False, f"样条创建失败，回退到Phoenix曲线: {spline_message}"
            
        # 混合曲线
        blended_curve = self.blend_with_phoenix(phoenix_curve, spline_curve, th_strength)
        
        # 检查混合后的单调性
        if not self.check_monotonicity(blended_curve):
            return phoenix_curve.copy(), False, "混合曲线非单调，自动回退到Phoenix曲线"
            
        # 成功使用样条
        status = f"样条曲线应用成功 (强度: {th_strength:.3f})"
        if spline_message:
            status += f" - {spline_message}"
            
        return blended_curve, True, status
        
    def get_spline_segments_info(self, th_nodes: List[float]) -> dict:
        """获取样条段信息
        
        Args:
            th_nodes: 样条节点
            
        Returns:
            info: 样条段信息字典
        """
        corrected_nodes, _ = self.validate_and_correct_nodes(th_nodes)
        full_nodes = [0.0] + corrected_nodes + [1.0]
        
        segments = []
        for i in range(len(full_nodes) - 1):
            segments.append({
                'start': full_nodes[i],
                'end': full_nodes[i + 1],
                'length': full_nodes[i + 1] - full_nodes[i]
            })
            
        return {
            'node_count': len(full_nodes),
            'segment_count': len(segments),
            'segments': segments,
            'corrected_nodes': corrected_nodes
        }


class SplineVisualizationHelper:
    """样条曲线可视化辅助类"""
    
    @staticmethod
    def generate_comparison_data(phoenix_curve: np.ndarray,
                               spline_curve: np.ndarray,
                               x_input: np.ndarray,
                               th_nodes: List[float]) -> dict:
        """生成对比可视化数据
        
        Args:
            phoenix_curve: Phoenix曲线
            spline_curve: 样条曲线
            x_input: 输入x坐标
            th_nodes: 样条节点
            
        Returns:
            visualization_data: 可视化数据字典
        """
        return {
            'x': x_input,
            'phoenix': phoenix_curve,
            'spline': spline_curve,
            'difference': spline_curve - phoenix_curve,
            'nodes': {
                'x': th_nodes,
                'y_phoenix': np.interp(th_nodes, x_input, phoenix_curve),
                'y_spline': np.interp(th_nodes, x_input, spline_curve)
            }
        }
        
    @staticmethod
    def compute_spline_statistics(phoenix_curve: np.ndarray,
                                spline_curve: np.ndarray) -> dict:
        """计算样条统计信息
        
        Args:
            phoenix_curve: Phoenix曲线
            spline_curve: 样条曲线
            
        Returns:
            stats: 统计信息字典
        """
        diff = spline_curve - phoenix_curve
        
        return {
            'max_deviation': float(np.max(np.abs(diff))),
            'mean_deviation': float(np.mean(np.abs(diff))),
            'rms_deviation': float(np.sqrt(np.mean(diff**2))),
            'positive_deviation_ratio': float(np.mean(diff > 0)),
            'spline_range': [float(np.min(spline_curve)), float(np.max(spline_curve))],
            'phoenix_range': [float(np.min(phoenix_curve)), float(np.max(phoenix_curve))]
        }